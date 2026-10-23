"""計測境界・SDK応答・失敗記録を検証する。実APIへの通信は行わない。"""

from collections import Counter
import json
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from app import day1_benchmark as benchmark
from app.day1_report import generate_report


class FakeClock:
    def __init__(self):
        self.value = 0.0
        self.waits = []

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.value += seconds

    def advance(self, seconds):
        self.value += seconds


def response(text="テスト用の回答", finish="STOP", usage=None, **extra):
    parts = [] if text is None else [types.Part.from_text(text=text)]
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=parts), finish_reason=finish,
        )], usage_metadata=usage, **extra,
    )


def manifest(models, interval=15.0, quotas=None):
    # このモードは実測レポート生成に使用できない。
    return {
        "schema_version": 1, "execution_mode": "offline_test",
        "run_id": "offline-benchmark-test", "models": models,
        "settings": {"min_interval_seconds": interval,
                     "quotas": quotas or {model: {} for model in models}},
    }


def read_rows(directory):
    return [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()]


def test_typed_sdk_usage_retains_thought_accounting_and_hides_thought_text():
    result = types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=[
                types.Part(text="内部のテスト文字列", thought=True),
                types.Part.from_text(text="表示する回答"),
            ]), finish_reason=types.FinishReason.STOP,
        )],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=5, candidates_token_count=7,
            thoughts_token_count=3, total_token_count=15,
        ),
        prompt_feedback={"block_reason": "BLOCKED_REASON_UNSPECIFIED"},
        model_version="sdk-fixture-model",
    )
    row = benchmark.parse_response(result)
    assert row["status"] == "ok" and row["text"] == "表示する回答"
    assert [row[key] for key in ("input_tokens", "output_tokens", "thought_tokens", "total_tokens")] == [5, 7, 3, 15]
    assert row["cached_tokens"] is None
    assert row["returned_model"] == "sdk-fixture-model"
    options = types.HttpOptions(timeout=120000, retry_options=types.HttpRetryOptions(attempts=1))
    assert options.retry_options.attempts == 1


def test_statuses_distinguish_blocked_empty_truncated_and_unexpected_finish():
    cases = [
        (response(), "ok"),
        (response(finish="MAX_TOKENS"), "truncated"),
        (response(finish="SAFETY"), "blocked"),
        (response(finish="RECITATION"), "blocked"),
        (response(finish="OTHER"), "error"),
        (response(text=None), "empty"),
        (types.GenerateContentResponse(prompt_feedback={"block_reason": "SAFETY"}), "blocked"),
        (types.GenerateContentResponse(), "empty"),
    ]
    for sdk_response, expected in cases:
        assert benchmark.parse_response(sdk_response)["status"] == expected


def test_missing_token_counts_are_unknown_and_real_zero_is_preserved():
    absent = benchmark.parse_response(response())
    assert all(absent[name] is None for name in benchmark.TOKEN_FIELDS)
    partial = benchmark.parse_response(response(usage=types.GenerateContentResponseUsageMetadata(
        prompt_token_count=11, candidates_token_count=0,
    )))
    assert partial["input_tokens"] == 11 and partial["output_tokens"] == 0
    assert partial["total_tokens"] is None and partial["thought_tokens"] is None
    for invalid in (-1, True, 1.5):
        assert benchmark.token_value(SimpleNamespace(prompt_token_count=invalid), "prompt_token_count") is None


def test_plan_covers_every_cell_and_interleaves_models_reproducibly():
    models = ["model-a", "model-b"]
    prompts = [{"id": name, "question": name} for name in ("general", "course", "grounded")]
    plan = benchmark.make_plan(models, prompts, repeats=3, seed=17)
    assert plan == benchmark.make_plan(models, prompts, repeats=3, seed=17)
    assert [row["order"] for row in plan] == list(range(1, 19))
    assert Counter((row["requested_model"], row["question_id"]) for row in plan) == {
        (model, prompt["id"]): 3 for model in models for prompt in prompts
    }
    for offset in range(0, len(plan), 2):
        pair = plan[offset:offset + 2]
        assert len({(row["trial"], row["question_id"]) for row in pair}) == 1
        assert {row["requested_model"] for row in pair} == set(models)
    assert {plan[offset]["requested_model"] for offset in range(0, len(plan), 2)} == set(models)


def test_latency_excludes_scheduled_wait_and_local_response_parsing(tmp_path, monkeypatch):
    clock = FakeClock()
    actual_parse = benchmark.parse_response

    def slow_local_parse(*args, **kwargs):
        parsed = actual_parse(*args, **kwargs)
        clock.advance(4.0)
        return parsed

    monkeypatch.setattr(benchmark, "parse_response", slow_local_parse)

    def request(model, question):
        clock.advance(2.0)
        return response()

    plan = benchmark.make_plan(["model-a"], [{"id": "q", "question": "Q"}], 2)
    directory = tmp_path / "latency"
    benchmark.run_benchmark(manifest(["model-a"]), plan, request, directory, clock=clock, sleep=clock.sleep)
    rows = read_rows(directory)
    assert [row["latency_seconds"] for row in rows] == [2.0, 2.0]
    assert [row["scheduled_wait_seconds"] for row in rows] == [0.0, 9.0]
    assert clock.waits == [9.0]


def test_per_model_rpm_spacing_is_applied_before_request_timer(tmp_path):
    clock = FakeClock()
    starts = []

    def request(model, question):
        starts.append(clock())
        clock.advance(1)
        return response()

    plan = benchmark.make_plan(["model-a"], [{"id": "q", "question": "Q"}], 2)
    directory = tmp_path / "rpm"
    metadata = manifest(["model-a"], interval=0, quotas={"model-a": {"rpm": 2}})
    benchmark.run_benchmark(metadata, plan, request, directory, clock=clock, sleep=clock.sleep)
    assert starts == pytest.approx([0, 33])
    assert read_rows(directory)[1]["latency_seconds"] == 1
    assert all(wait <= 30 for wait in clock.waits)


def test_429_preserves_failed_row_stops_only_that_model_and_redacts_secrets(tmp_path, capsys):
    clock = FakeClock()
    fake_key = "unit-test-key-" + "x" * 24
    api_key_shape = "AIza" + "x" * 35
    calls = []

    def request(model, question):
        calls.append(model)
        clock.advance(0.5)
        if model == "model-a":
            raise errors.ClientError(429, {"error": {"code": 429, "message": fake_key + api_key_shape, "status": "RESOURCE_EXHAUSTED"}})
        return response(text="回答 " + fake_key + " " + api_key_shape)

    models = ["model-a", "model-b"]
    plan = benchmark.make_plan(models, [{"id": "q", "question": "Q"}], 3)
    directory = tmp_path / "quota"
    finished = benchmark.run_benchmark(manifest(models), plan, request, directory, key=fake_key, clock=clock, sleep=clock.sleep)
    rows = read_rows(directory)
    assert Counter(calls) == {"model-a": 1, "model-b": 3}
    assert rows[0]["status"] == "error" and rows[0]["http_status"] == 429
    assert rows[0]["error_category"] == "rate_limit"
    assert finished["completed_requests"] == 4 and len(finished["skipped"]) == 2
    assert finished["status"] == "completed_with_skips"
    assert all(row["execution_mode"] == "offline_test" for row in rows)
    written = "".join(path.read_text(encoding="utf-8") for path in directory.iterdir()) + capsys.readouterr().out
    assert fake_key not in written and api_key_shape not in written
    assert "[REDACTED_API_KEY]" in written
    with pytest.raises(ValueError, match="live|offline_test"):
        generate_report(directory)
    assert not (directory / "answer.md").exists()


def test_503_is_distinct_from_quota_and_repeated_failures_stop_model(tmp_path):
    clock = FakeClock()
    calls = []

    def request(model, question):
        calls.append(model)
        clock.advance(0.5)
        raise errors.ServerError(503, {"error": {"code": 503, "message": "offline service fixture", "status": "UNAVAILABLE"}})

    plan = benchmark.make_plan(["model-a"], [{"id": "q", "question": "Q"}], 3)
    directory = tmp_path / "unavailable"
    finished = benchmark.run_benchmark(manifest(["model-a"]), plan, request, directory, clock=clock, sleep=clock.sleep)
    rows = read_rows(directory)
    assert len(calls) == 2 and len(rows) == 2
    assert all(row["http_status"] == 503 and row["error_category"] == "service_unavailable" for row in rows)
    assert finished["skipped"][0]["reason"] == "consecutive_service_unavailable"


def test_503_retry_after_is_honored_before_next_attempt(tmp_path):
    clock = FakeClock()
    starts = []

    def request(model, question):
        starts.append(clock())
        clock.advance(1)
        if len(starts) == 1:
            raise errors.ServerError(
                503,
                {"error": {"code": 503, "message": "offline fixture", "status": "UNAVAILABLE"}},
                response=SimpleNamespace(headers={"retry-after": "45"}),
            )
        return response()

    plan = benchmark.make_plan(["model-a"], [{"id": "q", "question": "Q"}], 2)
    directory = tmp_path / "retry_after"
    benchmark.run_benchmark(manifest(["model-a"]), plan, request, directory, clock=clock, sleep=clock.sleep)
    rows = read_rows(directory)
    assert starts == [0, 46]
    assert rows[0]["retry_after_seconds"] == 45
    assert rows[1]["scheduled_wait_seconds"] == 45
    assert rows[1]["latency_seconds"] == 1


def test_keyboard_interrupt_preserves_current_attempt(tmp_path):
    clock = FakeClock()

    def request(model, question):
        clock.advance(0.25)
        raise KeyboardInterrupt

    plan = benchmark.make_plan(["model-a"], [{"id": "q", "question": "Q"}], 2)
    directory = tmp_path / "interrupted"
    finished = benchmark.run_benchmark(manifest(["model-a"]), plan, request, directory, clock=clock, sleep=clock.sleep)
    rows = read_rows(directory)
    assert finished["status"] == "interrupted" and finished["ended_at_utc"]
    assert len(rows) == 1 and rows[0]["error_category"] == "interrupted"
    assert rows[0]["latency_seconds"] == 0.25
