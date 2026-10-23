"""保存済みの2回分を再集計する。API呼出しも元記録の変更も行わない。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN_NAMES = ("day1_run", "day1_flash_retry")
FIELDS = ("latency_seconds", "input_tokens", "output_tokens", "thought_tokens", "total_tokens", "cached_tokens")


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def jst(text: str) -> str:
    return datetime.fromisoformat(text).astimezone(timezone(timedelta(hours=9))).isoformat(timespec="milliseconds")


def main() -> None:
    output = ROOT / "results" / "analysis"
    sources, runs, rows, planned, skipped, seen = {}, [], [], defaultdict(int), defaultdict(int), set()
    for name in RUN_NAMES:
        folder = ROOT / "results" / name
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        current = [json.loads(line) for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if manifest["execution_mode"] != "live" or len(current) != manifest["completed_requests"]:
            raise ValueError("実行種別または記録件数がmanifestと一致しません")
        if len(current) + len(manifest["skipped"]) != manifest["planned_requests"]:
            raise ValueError("計画件数と実送信・未送信の合計が一致しません")
        for record in current:
            identity = (record["run_id"], record["order"])
            if record["run_id"] != manifest["run_id"] or identity in seen:
                raise ValueError("run_idの不一致または重複した試行があります")
            seen.add(identity)
            if record["execution_mode"] != "live":
                raise ValueError("実測以外の記録は集計しません")
            for field in FIELDS:
                value = record.get(field)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                    raise ValueError(f"不正な計測値: {field}")
        for model in manifest["models"]:
            for prompt in manifest["prompts"]:
                planned[model, prompt["id"]] += manifest["repeats"]
        for record in manifest["skipped"]:
            skipped[record["requested_model"], record["question_id"]] += 1
        runs.append({"run_name": name, "run_id": manifest["run_id"], "started_jst": jst(manifest["started_at_utc"]), "ended_jst": jst(manifest["ended_at_utc"]), "planned": manifest["planned_requests"], "attempted": len(current), "ok": sum(r["status"] == "ok" for r in current), "http_503": sum(r.get("http_status") == 503 for r in current), "http_429": sum(r.get("http_status") == 429 for r in current), "skipped": len(manifest["skipped"])})
        rows.extend(current)
        for filename in ("manifest.json", "results.jsonl"):
            sources[f"results/{name}/{filename}"] = hashlib.sha256((folder / filename).read_bytes()).hexdigest()

    conditions = []
    for (model, question), count in planned.items():
        selected = [r for r in rows if r["requested_model"] == model and r["question_id"] == question]
        successful = [r for r in selected if r["status"] == "ok"]
        row = {"requested_model": model, "question_id": question, "planned": count, "attempted": len(selected), "ok": len(successful), "http_503": sum(r.get("http_status") == 503 for r in selected), "http_429": sum(r.get("http_status") == 429 for r in selected), "skipped": skipped[model, question]}
        for field in FIELDS:
            values = [r[field] for r in successful if r.get(field) is not None]
            row[field + "_n"] = len(values)
            row[field + "_values"] = json.dumps(values)
            row[field + "_median"] = statistics.median(values) if values else None
            row[field + "_min"] = min(values) if values else None
            row[field + "_max"] = max(values) if values else None
        conditions.append(row)

    characters = []
    for row in rows:
        text = row.get("text", "")
        claimed = re.search(r"[（(](\d+)(?:文字|字)[）)]\s*$", text)
        actual = len(text) if text else None
        count = int(claimed.group(1)) if claimed else None
        characters.append({"run_id": row["run_id"], "order": row["order"], "requested_model": row["requested_model"], "question_id": row["question_id"], "trial": row["trial"], "status": row["status"], "raw_characters": actual, "claimed_characters": count, "within_200_characters": actual <= 200 if actual is not None else None, "claim_matches_raw_length": count == actual if count is not None else None})
    totals = {key: sum(r[key] for r in runs) for key in ("planned", "attempted", "ok", "http_503", "http_429", "skipped")}
    successful = [r for r in rows if r["status"] == "ok"]
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        values = [r[field] for r in successful if r.get(field) is not None]
        totals["success_" + field] = sum(values) if values else None
        totals["success_" + field + "_n"] = len(values)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "run_summary.csv", runs)
    write_csv(output / "condition_summary.csv", conditions)
    write_csv(output / "character_counts.csv", characters)
    (output / "analysis_summary.json").write_text(json.dumps({"totals": totals, "source_sha256": sources, "character_count_method": "Python len(text): 改行・Markdown記号・末尾の自己申告を含むUnicode文字列長", "semantic_review": "ai_assisted_quality_review.csvはChatGPTによる参照資料との照合。数値集計とは別の評価。"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"集計完了: 実送信{totals['attempted']}件、成功{totals['ok']}件、503={totals['http_503']}件、未送信{totals['skipped']}件")


if __name__ == "__main__":
    main()
