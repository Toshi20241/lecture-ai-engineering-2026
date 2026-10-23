"""Day1: repeatable Gemini measurements. No API requests unless --run is given.

Usage (from the course repository root):
    uv run python -m app.day1_benchmark --include-grounded
    uv run python -m app.day1_benchmark --include-grounded --run

The first command only displays the plan. Each logical trial is one SDK attempt.
Offline tests are explicitly labelled and cannot generate a live submission.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import re
import subprocess
import time
from typing import Any, Callable
import uuid


COURSE_COMMIT = "662cddd52724b330f3f4cd15123d1e55cb474874"
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
ROOT = Path(__file__).resolve().parents[1]
TOKEN_FIELDS = {
    "input_tokens": "prompt_token_count",
    "output_tokens": "candidates_token_count",
    "thought_tokens": "thoughts_token_count",
    "total_tokens": "total_token_count",
    "cached_tokens": "cached_content_token_count",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def dump_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def make_prompts(glossary_path: Path, include_grounded: bool) -> list[dict]:
    glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
    reference = next(t["definition"] for t in glossary["terms"] if t["id"] == "homework-levels")
    question = "AIE2026の講座の宿題にある「3レベル」の名称と、修了に必要な取り組み数を200字以内で説明してください。"
    prompts = [
        {"id": "general_lora", "label": "一般知識: LoRA", "question": "LoRAとは何ですか。元の重みと学習するパラメータを区別し、200字以内で説明してください。", "rubric": "元の重みを固定し、更新を低ランク行列で表し、その追加パラメータを学習する。簡潔さと内容の正しさを人が確認する。"},
        {"id": "course_ungrounded", "label": "講座固有: 参照情報なし", "question": question, "reference": reference, "rubric": "基礎・実装・発展の名称と、どれか1つ以上で完了という条件。根拠のない追加事項と不明表明も記録する。"},
    ]
    if include_grounded:
        prompts.append({"id": "course_grounded", "label": "講座固有: 参照情報あり", "question": question + "\n\n次の参考情報に基づいて回答してください。\n" + reference, "reference": reference, "rubric": prompts[1]["rubric"]})
    return prompts


def make_plan(models: list[str], prompts: list[dict], repeats: int, seed: int = 20261022) -> list[dict]:
    """Rotate model order and shuffle questions within repeat blocks."""
    if not models or len(set(models)) != len(models):
        raise ValueError("モデル名は重複させず、1つ以上指定してください。")
    if repeats < 1:
        raise ValueError("repeats は1以上が必要です。")
    rng = random.Random(seed)
    plan: list[dict] = []
    for trial in range(1, repeats + 1):
        block = list(prompts)
        rng.shuffle(block)
        for index, prompt in enumerate(block):
            offset = (trial - 1 + index) % len(models)
            ordered = models[offset:] + models[:offset]
            for model in ordered:
                plan.append({"order": len(plan) + 1, "trial": trial, "question_id": prompt["id"], "question": prompt["question"], "requested_model": model})
    return plan


def load_quotas(path: Path | None, models: list[str]) -> dict:
    supplied = json.loads(path.read_text(encoding="utf-8")) if path else {}
    quotas = {}
    for model in models:
        value = supplied.get(model, {})
        quotas[model] = {key: value.get(key) for key in ("rpm", "tpm", "rpd")}
        quotas[model]["source"] = value.get("source", "未確認: AI Studioの当該プロジェクトで確認する")
        for key in ("rpm", "tpm", "rpd"):
            number = quotas[model][key]
            if number is not None and (isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0):
                raise ValueError(f"{model} の {key} は正の数か null にしてください。")
    return quotas


def token_value(usage: Any, field: str) -> int | None:
    value = getattr(usage, field, None) if usage is not None else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def redact(text: str, key: str = "") -> str:
    if key:
        text = text.replace(key, "[REDACTED_API_KEY]")
    return re.sub(r"AIza[0-9A-Za-z_-]{30,}", "[REDACTED_API_KEY]", text)


def parse_response(response: Any, key: str = "") -> dict:
    candidates = getattr(response, "candidates", None) or []
    candidate = candidates[0] if candidates else None
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    text = "".join(getattr(part, "text", "") or "" for part in parts if not getattr(part, "thought", False))
    raw_finish = getattr(candidate, "finish_reason", None)
    finish = getattr(raw_finish, "value", raw_finish)
    feedback = getattr(response, "prompt_feedback", None)
    block = getattr(feedback, "block_reason", None)
    block = getattr(block, "value", block)
    blocked = block not in (None, "BLOCKED_REASON_UNSPECIFIED") or finish in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION", "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION"}
    if blocked:
        status = "blocked"
    elif finish == "MAX_TOKENS":
        status = "truncated"
    elif not text.strip():
        status = "empty"
    elif finish not in (None, "STOP"):
        status = "error"
    else:
        status = "ok"
    usage = getattr(response, "usage_metadata", None)
    cleaned = redact(text, key)
    return {
        "status": status,
        "text": cleaned,
        "text_redacted": cleaned != text,
        "finish_reason": finish,
        "returned_model": getattr(response, "model_version", None),
        "prompt_block_reason": block,
        "error_category": "unexpected_finish_reason" if status == "error" else None,
        **{name: token_value(usage, field) for name, field in TOKEN_FIELDS.items()},
    }


def parse_error(error: Exception) -> dict:
    """Only retain structured status, never raw server messages or request URLs."""
    code = getattr(error, "code", None)
    code = code if isinstance(code, int) else None
    category = {429: "rate_limit", 503: "service_unavailable", 400: "invalid_request", 401: "authentication_or_access", 403: "authentication_or_access", 404: "model_not_found"}.get(code, "api_error" if code else "network_or_client_error")
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw_retry = headers.get("retry-after")
    delay = None
    if raw_retry:
        try:
            delay = max(0.0, float(raw_retry))
        except (ValueError, TypeError):
            try:
                delay = max(0.0, (parsedate_to_datetime(str(raw_retry)) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    if delay is not None and not math.isfinite(delay):
        delay = None
    return {"status": "error", "http_status": code, "error_category": category, "retry_after_seconds": delay}


def run_benchmark(
    manifest: dict,
    plan: list[dict],
    request: Callable[[str, str], Any],
    run_dir: Path,
    *,
    key: str = "",
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Transport is injected for tests; their manifest must say offline_test."""
    if manifest.get("execution_mode") not in {"live", "offline_test"}:
        raise ValueError("execution_mode must be live or offline_test")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest.update(status="running", completed_requests=0, skipped=[], ended_at_utc=None)
    dump_json(run_dir / "manifest.json", manifest)
    dump_json(run_dir / "plan.json", plan)
    blocked_models: dict[str, str] = {}
    server_failures: dict[str, int] = {}
    previous_start: float | None = None
    previous_by_model: dict[str, float] = {}
    not_before: dict[str, float] = {}
    settings = manifest["settings"]
    try:
        with (run_dir / "results.jsonl").open("x", encoding="utf-8", buffering=1) as out:
            for item in plan:
                model = item["requested_model"]
                if model in blocked_models:
                    manifest["skipped"].append({**{k: item[k] for k in ("order", "trial", "question_id", "requested_model")}, "reason": blocked_models[model]})
                    continue
                now = clock()
                wait = max(0.0, settings["min_interval_seconds"] - (now - previous_start)) if previous_start is not None else 0.0
                rpm = settings["quotas"].get(model, {}).get("rpm")
                if rpm and model in previous_by_model:
                    wait = max(wait, 60.0 / rpm * 1.10 - (now - previous_by_model[model]))
                wait = max(wait, not_before.get(model, now) - now)
                remaining = wait
                while remaining > 0:
                    chunk = min(remaining, 30.0)
                    sleep(chunk)
                    remaining -= chunk
                row = {
                    **item, "execution_mode": manifest["execution_mode"], "run_id": manifest["run_id"],
                    "started_at_utc": now_utc(), "scheduled_wait_seconds": round(wait, 6),
                    "http_status": None, "error_category": None, "retry_after_seconds": None,
                    "text": "", "returned_model": None, "finish_reason": None,
                    **{name: None for name in TOKEN_FIELDS},
                }
                previous_start = previous_by_model[model] = clock()
                interrupted = False
                try:
                    response = request(model, item["question"])
                    # Stop the request timer before local parsing, redaction or disk writes.
                    row["latency_seconds"] = round(clock() - previous_start, 6)
                    row.update(parse_response(response, key))
                except KeyboardInterrupt:
                    row.update(status="error", error_category="interrupted", latency_seconds=round(clock() - previous_start, 6))
                    interrupted = True
                except Exception as error:
                    row["latency_seconds"] = round(clock() - previous_start, 6)
                    row.update(parse_error(error))
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                manifest["completed_requests"] += 1
                code = row["http_status"]
                if code in {400, 401, 403, 404, 429}:
                    blocked_models[model] = row["error_category"]
                if code == 503:
                    server_failures[model] = server_failures.get(model, 0) + 1
                    if row["retry_after_seconds"] is not None:
                        not_before[model] = clock() + row["retry_after_seconds"]
                    if server_failures[model] >= 2:
                        blocked_models[model] = "consecutive_service_unavailable"
                else:
                    server_failures[model] = 0
                print(f"[{item['order']}/{len(plan)}] {model} {item['question_id']} #{item['trial']} {row['status']} {row['latency_seconds']:.3f}s", flush=True)
                dump_json(run_dir / "manifest.json", manifest)
                if interrupted:
                    manifest["status"] = "interrupted"
                    break
            else:
                manifest["status"] = "completed_with_skips" if manifest["skipped"] else "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
    finally:
        manifest["ended_at_utc"] = now_utc()
        dump_json(run_dir / "manifest.json", manifest)
    return manifest


def local_provenance() -> dict:
    hashes = {}
    for relative in ("app/day1_benchmark.py", "app/day1_report.py", "app/ask.py", "data/glossary.json", "uv.lock"):
        path = ROOT / relative
        if path.is_file():
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"local_git_commit": revision, "source_sha256": hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Day1 比較実験: --run を付けるまでAPIは呼びません。")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-grounded", action="store_true", help="用語集を渡した条件も追加（2モデルなら合計18回）")
    parser.add_argument("--run", action="store_true", help="Gemini APIを実際に呼び出す")
    parser.add_argument("--output", type=Path, help="新しい実験フォルダ。既存フォルダには追記しません。")
    parser.add_argument("--answer", type=Path, help="提出下書きの保存先。既存の手書き解答は保護します。")
    parser.add_argument("--quota-file", type=Path)
    parser.add_argument("--max-requests", type=int, default=18)
    parser.add_argument("--min-interval", type=float, default=15.0)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    for model in args.models:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,150}", model):
            parser.error("モデル名の書式が不正です。")
    if not math.isfinite(args.min_interval) or args.min_interval < 0:
        parser.error("min-interval は0以上の有限値が必要です。")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0 or args.max_output_tokens < 1:
        parser.error("timeout-seconds と max-output-tokens は正の値が必要です。")
    try:
        prompts = make_prompts(ROOT / "data/glossary.json", args.include_grounded)
        plan = make_plan(args.models, prompts, args.repeats, args.seed)
        quotas = load_quotas(args.quota_file, args.models)
    except (ValueError, KeyError, OSError, StopIteration) as error:
        parser.error(str(error))
    if len(plan) > args.max_requests:
        parser.error(f"計画は{len(plan)}回で max-requests={args.max_requests} を超えます。回数を減らすか、利用枠を確認して上限を変更してください。")
    calls_each = len(prompts) * args.repeats
    for model in args.models:
        if quotas[model]["rpd"] is not None and calls_each > quotas[model]["rpd"]:
            parser.error(f"{model}: 計画{calls_each}回が記入されたRPDを超えます。")
    print(f"計画: {len(args.models)}モデル × {len(prompts)}質問 × {args.repeats}回 = {len(plan)}リクエスト（モデルあたり{calls_each}回）")
    print(f"送信間隔: 全体{args.min_interval:g}秒以上。quota-fileにRPMを記入したモデルは追加間隔を確保。SDKの自動再試行は無効。")
    print("日次の残り枠・TPM・同じプロジェクトでの他の利用は、AI Studioで確認してください。")
    if not args.run:
        for item in plan:
            print(f"{item['order']:02}: {item['requested_model']} / {item['question_id']} / repeat {item['trial']}")
        print("APIは呼び出していません。実測は同じコマンドの末尾に --run を追加します。")
        return
    from google import genai
    from google.genai import types
    from app.config import get_api_key

    key = get_api_key()
    if args.answer and args.answer.exists():
        from app.day1_report import AUTO_MARKER
        if AUTO_MARKER not in args.answer.read_text(encoding="utf-8"):
            parser.error("answer に既存の手書き解答があります。別名（answer_generated.mdなど）を指定してください。")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_dir = args.output or ROOT / "lectures/01/results" / run_id
    manifest = {
        "schema_version": 1, "execution_mode": "live", "run_id": run_id,
        "started_at_utc": now_utc(), "models": args.models, "repeats": args.repeats,
        "prompts": prompts, "planned_requests": len(plan), "course_commit": COURSE_COMMIT,
        "course_repository": "https://github.com/matsuolab/lecture-ai-engineering-2026",
        "versions": {"python": platform.python_version(), "google_genai": importlib.metadata.version("google-genai")},
        "settings": {"max_output_tokens": args.max_output_tokens, "min_interval_seconds": args.min_interval,
                     "sdk_attempts": 1, "timeout_ms": int(args.timeout_seconds * 1000), "temperature": "model_default",
                     "thinking_config": "model_default", "seed_for_request_order_only": args.seed, "quotas": quotas},
        **local_provenance(),
    }
    options = types.HttpOptions(timeout=manifest["settings"]["timeout_ms"], retry_options=types.HttpRetryOptions(attempts=1))
    config = types.GenerateContentConfig(max_output_tokens=args.max_output_tokens)
    with genai.Client(api_key=key, http_options=options) as client:
        def request(model: str, question: str) -> Any:
            return client.models.generate_content(model=model, contents=question, config=config)
        try:
            run_benchmark(manifest, plan, request, run_dir, key=key)
        except FileExistsError:
            parser.error("output フォルダが既にあります。新しい名前を指定してください。")
    from app.day1_report import generate_report
    try:
        paths = generate_report(run_dir, args.answer)
    except (ValueError, OSError) as error:
        print(f"計測記録は {run_dir} に保存済みです。集計: {error}")
        raise SystemExit(1) from None
    print("\n計測と集計の保存先: " + str(run_dir))
    print("実際の回答を読み、手動評価と考察を記入してから提出してください。")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
