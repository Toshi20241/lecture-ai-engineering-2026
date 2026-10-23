"""集計の誤り・実測とテストの取り違えを検出する。API は使わない。"""

import csv
import json
from copy import deepcopy

import pytest

from app.day1_report import (
    AUTO_MARKER, generate_report, load_run, measurement_readiness, summarize,
)


def manifest(mode="live"):
    # このfixtureは一時ディレクトリのテストだけに使い、提出結果に含めない。
    return {
        "schema_version": 1, "execution_mode": mode, "run_id": "unit-test-only",
        "models": ["model-a", "model-b"], "repeats": 3,
        "prompts": [{"id": "general", "label": "一般", "question": "Q1"}, {"id": "course", "label": "講座", "question": "Q2"}],
        "settings": {"max_output_tokens": 512, "temperature": "model_default", "sdk_attempts": 1},
        "planned_requests": 12, "status": "completed",
    }


def row(order=1, trial=1, model="model-a", question="general", **overrides):
    result = {
        "execution_mode": "live", "run_id": "unit-test-only", "order": order,
        "trial": trial, "requested_model": model, "question_id": question,
        "question": "test fixture", "status": "ok", "latency_seconds": 2.0,
        "input_tokens": 10, "output_tokens": 20, "total_tokens": None,
        "thought_tokens": None, "cached_tokens": None, "text": "fixture response",
    }
    result.update(overrides)
    return result


def write_run(tmp_path, meta, rows):
    (tmp_path / "manifest.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_missing_usage_is_missing_not_zero():
    results = summarize(manifest(), [row(input_tokens=None, output_tokens=None)])
    assert results[0]["input_tokens_n"] == 0
    assert results[0]["input_tokens_median"] is None
    assert results[0]["output_tokens_values"] == []
    assert not measurement_readiness(manifest(), [row(input_tokens=None)])["advanced_measurements"]


def test_failures_and_truncation_not_in_success_distribution():
    rows = [
        row(1, 1, latency_seconds=1), row(2, 2, latency_seconds=3),
        row(3, 3, latency_seconds=100, status="error", http_status=429),
        row(4, 4, latency_seconds=200, status="truncated"),
        row(5, 5, latency_seconds=300, status="error", http_status=503),
    ]
    s = summarize(manifest(), rows)[0]
    assert (s["attempts"], s["ok"], s["error"], s["truncated"]) == (5, 2, 2, 1)
    assert (s["latency_seconds_median"], s["latency_seconds_min"], s["latency_seconds_max"]) == (2, 1, 3)
    assert s["http_429"] == 1 and s["http_503"] == 1


@pytest.mark.parametrize("where", ["manifest", "row"])
def test_offline_data_refused(tmp_path, where):
    meta, sample = manifest(), row()
    (meta if where == "manifest" else sample)["execution_mode"] = "offline_test"
    write_run(tmp_path, meta, [sample])
    with pytest.raises(ValueError, match="live|実行モード"):
        generate_report(tmp_path)
    assert not (tmp_path / "answer.md").exists()


@pytest.mark.parametrize("override", [{"run_id": "other"}, {"latency_seconds": float("nan")}, {"input_tokens": -1}])
def test_invalid_provenance_and_numbers_refused(tmp_path, override):
    write_run(tmp_path, manifest(), [row(**override)])
    with pytest.raises(ValueError):
        load_run(tmp_path)


def test_duplicate_order_refused(tmp_path):
    write_run(tmp_path, manifest(), [row(), row(trial=2)])
    with pytest.raises(ValueError, match="order"):
        load_run(tmp_path)


def test_arbitrary_answer_is_not_overwritten(tmp_path):
    write_run(tmp_path, manifest(), [row()])
    answer = tmp_path / "answer.md"
    answer.write_text("自分で記入した考察", encoding="utf-8")
    with pytest.raises(ValueError, match="上書き"):
        generate_report(tmp_path)
    assert answer.read_text(encoding="utf-8") == "自分で記入した考察"


def test_readiness_requires_shared_questions_and_all_measurement_fields():
    rows = [row(1, model="model-a", question="general"), row(2, model="model-a", question="course"), row(3, model="model-b", question="general"), row(4, model="model-b", question="course")]
    assert measurement_readiness(manifest(), rows)["advanced_measurements"]
    altered = deepcopy(rows)
    altered[-1]["output_tokens"] = None
    assert not measurement_readiness(manifest(), altered)["advanced_measurements"]
    assert not measurement_readiness(manifest(), rows)["basic_three_repeats"]


def test_regeneration_preserves_quality_reviews_and_reports_missing_data(tmp_path):
    write_run(tmp_path, manifest(), [row(input_tokens=None)])
    output = generate_report(tmp_path)
    report = (tmp_path / "answer.md").read_text(encoding="utf-8")
    assert AUTO_MARKER in report
    assert "未取得 (0/1)" in report
    assert "発展に向けた測定途中" in report
    review = tmp_path / "manual_quality_review.csv"
    with review.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, data = reader.fieldnames, list(reader)
    assert data[0]["factual_agreement_0_1_2"] == ""
    data[0]["review_note"] = "自分で照合した記録"
    with review.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)
    generate_report(tmp_path)
    assert "自分で照合した記録" in review.read_text(encoding="utf-8")
    assert set(output) == {"answer", "summary", "responses", "quality_review", "chart"}
