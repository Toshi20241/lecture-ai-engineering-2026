"""Day1: 実 API 記録だけを集計し、再確認できる解答の下書きを生成する。

使い方: uv run python -m app.day1_report lectures/01/results/<run_id>
合成データによるモデル性能の報告を防ぐため、offline_test は受け付けない。
"""

from __future__ import annotations

import argparse
import csv
import html
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

AUTO_MARKER = "<!-- AIE2026-DAY1-AUTO-REPORT-v1 -->"
METRICS = (
    "latency_seconds", "input_tokens", "output_tokens", "thought_tokens",
    "total_tokens", "cached_tokens",
)
TOKEN_METRICS = METRICS[1:]
STATUSES = {"ok", "empty", "truncated", "blocked", "error"}
QUALITY_FIELDS = [
    "order", "model", "question_id", "trial", "status", "reference_source",
    "factual_agreement_0_1_2", "unsupported_claims", "uncertainty_handling",
    "format_compliance", "review_note",
]


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """実行モード、行の同一性、数値を検証する。証跡の真正性を保証する機能ではない。"""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("対応していない記録形式です。schema_version=1 が必要です。")
    if manifest.get("execution_mode") != "live":
        raise ValueError("実 API の live 記録のみを集計します。offline_test は性能測定ではありません。")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id がありません。")
    models = manifest.get("models", [])
    prompts = manifest.get("prompts", [])
    if not models or len(models) != len(set(models)):
        raise ValueError("モデル一覧が空か重複しています。")
    prompt_ids = [p["id"] for p in prompts]
    if not prompt_ids or len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("質問一覧が空か重複しています。")
    rows = []
    orders: set[int] = set()
    conditions: set[tuple[Any, ...]] = set()
    for line_no, line in enumerate((run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("execution_mode") != "live" or row.get("run_id") != run_id:
            raise ValueError(f"行 {line_no}: 実行モードまたは run_id が一致しません。")
        order = row.get("order")
        if not isinstance(order, int) or isinstance(order, bool) or order < 1 or order in orders:
            raise ValueError(f"行 {line_no}: order が無効か重複しています。")
        orders.add(order)
        if row.get("requested_model") not in models or row.get("question_id") not in prompt_ids:
            raise ValueError(f"行 {line_no}: 未定義のモデルまたは質問です。")
        trial = row.get("trial")
        if not isinstance(trial, int) or isinstance(trial, bool) or trial < 1:
            raise ValueError(f"行 {line_no}: trial が無効です。")
        condition = (row["requested_model"], row["question_id"], trial)
        if condition in conditions:
            raise ValueError(f"行 {line_no}: 同じ条件・試行が重複しています。")
        conditions.add(condition)
        if row.get("status") not in STATUSES:
            raise ValueError(f"行 {line_no}: 未定義の status です。")
        for key in METRICS:
            value = row.get(key)
            if value is not None and (not _number(value) or value < 0):
                raise ValueError(f"行 {line_no}: {key} の数値が無効です。")
            if key in TOKEN_METRICS and value is not None and not isinstance(value, int):
                raise ValueError(f"行 {line_no}: {key} は整数または null が必要です。")
        rows.append(row)
    if not rows:
        raise ValueError("API 呼出し記録がありません。モデル比較結果はまだ生成できません。")
    return manifest, sorted(rows, key=lambda r: r["order"])


def summarize(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """成功のみの分布と、全呼出しの状態内訳を条件ごとに集計する。"""
    summaries = []
    for model in manifest["models"]:
        for prompt in manifest["prompts"]:
            group = [r for r in rows if r["requested_model"] == model and r["question_id"] == prompt["id"]]
            ok = [r for r in group if r["status"] == "ok"]
            counts = Counter(r["status"] for r in group)
            result: dict[str, Any] = {
                "model": model, "question_id": prompt["id"], "question_label": prompt.get("label", prompt["id"]),
                "attempts": len(group), **{s: counts[s] for s in sorted(STATUSES)},
                "http_429": sum(r.get("http_status") == 429 for r in group),
                "http_503": sum(r.get("http_status") == 503 for r in group),
            }
            for key in METRICS:
                values = [r[key] for r in ok if _number(r.get(key))]
                result[key + "_n"] = len(values)
                result[key + "_values"] = values
                result[key + "_median"] = statistics.median(values) if values else None
                result[key + "_min"] = min(values) if values else None
                result[key + "_max"] = max(values) if values else None
            summaries.append(result)
    return summaries


def measurement_readiness(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """測定条件の充足のみ判定。本人の考察、提出、成績の判定は行わない。"""
    complete = [r for r in rows if r["status"] == "ok" and all(_number(r.get(k)) for k in ("latency_seconds", "input_tokens", "output_tokens"))]
    questions = {m: {r["question_id"] for r in complete if r["requested_model"] == m} for m in manifest["models"]}
    matched = [(a, b, sorted(questions[a] & questions[b])) for a, b in itertools.combinations(manifest["models"], 2) if len(questions[a] & questions[b]) >= 2]
    basic_complete = len(manifest["models"]) >= 2 and all(
        len({r["trial"] for r in rows if r["status"] == "ok" and r["requested_model"] == model and r["question_id"] == prompt["id"]}) >= 3
        for model in manifest["models"] for prompt in manifest["prompts"]
    )
    return {"advanced_measurements": bool(matched), "matched_pairs": matched, "basic_three_repeats": basic_complete}


def _fmt(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "未取得"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{decimals}f}"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _stat_cell(summary: dict[str, Any], metric: str) -> str:
    n = summary[metric + "_n"]
    if not n:
        return f"未取得 (0/{summary['ok']})"
    return f"{_fmt(summary[metric + '_median'])} [{_fmt(summary[metric + '_min'])}, {_fmt(summary[metric + '_max'])}] ({n}/{summary['ok']})"


def _write_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = list(summaries[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in summary.items()})


def _quality_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """再集計で本人の評価記入を消さず、新たな呼出し行だけ追記する。"""
    existing = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != QUALITY_FIELDS:
                raise ValueError("manual_quality_review.csv の列が変更されています。既存ファイルを保護するため停止します。")
            existing = {int(r["order"]): r for r in reader}
    # 検査を完了してから書き始める。途中で不一致があっても既存の記入を失わない。
    for row in rows:
        if row["order"] in existing:
            old = existing[row["order"]]
            if old["model"] != row["requested_model"] or old["question_id"] != row["question_id"] or int(old["trial"]) != row["trial"]:
                raise ValueError("既存の品質評価と測定行が一致しません。")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=QUALITY_FIELDS)
        writer.writeheader()
        for row in rows:
            base = {
                "order": row["order"], "model": row["requested_model"], "question_id": row["question_id"],
                "trial": row["trial"], "status": row["status"], "reference_source": "参照資料と照合して記入",
            }
            if row["order"] in existing:
                base.update(existing[row["order"]])
            writer.writerow(base)


def _responses(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    parts = ["# 応答本文と試行記録", "", f"Run ID: `{manifest['run_id']}`", "", "`results.jsonl` に対応する全呼出し。回答内容の正しさは本人が参照資料と照合する。", ""]
    for row in rows:
        parts += [
            f"## 呼出し {row['order']}: {_md(row['requested_model'])} / {_md(row['question_id'])} / 試行 {row['trial']}", "",
            f"- 状態: `{row['status']}` / HTTP: {_fmt(row.get('http_status'))} / 終了理由: `{row.get('finish_reason') or '未取得'}`",
            f"- 時間: {_fmt(row.get('latency_seconds'))} 秒 / 入力: {_fmt(row.get('input_tokens'))} / 出力: {_fmt(row.get('output_tokens'))} tokens",
            f"- 返却モデル: `{row.get('returned_model') or '未取得'}`", "", "### 質問", "",
            *["> " + line for line in row.get("question", "").splitlines()], "", "### 回答", "",
            *["> " + line for line in (row.get("text") or "（テキスト応答なし）").splitlines()], "",
        ]
        if row.get("error_category"):
            parts += [f"エラー分類: `{row['error_category']}`", ""]
    return "\n".join(parts) + "\n"


def _chart(summaries: list[dict[str, Any]]) -> str:
    data = [s for s in summaries if s["latency_seconds_n"]]
    height = 110 + max(1, len(data)) * 54
    width, left, plot_width = 1040, 380, 470
    maximum = max((s["latency_seconds_max"] for s in data), default=1) or 1
    esc = html.escape
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="成功応答の時間。棒は中央値、線は最小値と最大値。">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<g font-family="sans-serif" fill="#0f172a">',
        '<text x="24" y="30" font-size="20" font-weight="bold">Response time by model and prompt</text>',
        '<text x="24" y="53" font-size="13">Successful responses only; bar: median, whisker: observed min-max (seconds).</text>',
    ]
    colors = ["#0f766e", "#1d4ed8", "#7c3aed", "#b45309"]
    model_colors = {m: colors[i % len(colors)] for i, m in enumerate(dict.fromkeys(s["model"] for s in data))}
    for i, s in enumerate(data):
        y = 88 + i * 54
        scale = plot_width / maximum
        median, lo, hi = (s["latency_seconds_" + k] * scale for k in ("median", "min", "max"))
        label = esc(f"{s['model']} / {s['question_id']}")
        elements += [
            f'<text x="24" y="{y + 4}" font-size="13">{label}</text>',
            f'<rect x="{left}" y="{y - 11}" width="{max(median, 0.5):.2f}" height="22" rx="3" fill="{model_colors[s["model"]]}" opacity="0.85"/>',
            f'<line x1="{left + lo:.2f}" x2="{left + hi:.2f}" y1="{y}" y2="{y}" stroke="#0f172a" stroke-width="2"/>',
            f'<line x1="{left + lo:.2f}" x2="{left + lo:.2f}" y1="{y - 6}" y2="{y + 6}" stroke="#0f172a"/>',
            f'<line x1="{left + hi:.2f}" x2="{left + hi:.2f}" y1="{y - 6}" y2="{y + 6}" stroke="#0f172a"/>',
            f'<text x="{left + plot_width + 14}" y="{y + 4}" font-size="13">{s["latency_seconds_median"]:.3f}s; n={s["latency_seconds_n"]}</text>',
        ]
    if not data:
        elements.append('<text x="24" y="92" font-size="15">No successful latency measurements.</text>')
    elements += ["</g>", "</svg>"]
    return "\n".join(elements) + "\n"


def build_answer(manifest: dict[str, Any], rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> str:
    readiness = measurement_readiness(manifest, rows)
    settings = manifest.get("settings", {})
    states = Counter(r["status"] for r in rows)
    achieved = readiness["advanced_measurements"]
    level_text = "発展（複数モデル・共通質問の実測あり。本人の考察を追記して提出）" if achieved else "発展に向けた測定途中（比較に必要な成功データが未充足）"
    parts = [
        AUTO_MARKER, "# Day1: 応答時間・トークン数と回答内容の比較", "",
        "> 実 API 記録から生成した下書き。回答本文の照合と、自分で確かめた考察を追記して提出する。", "",
        "## 1. 取り組みと現在の結果", "",
        f"- 取り組み: **{level_text}**", f"- 実行 ID: `{manifest['run_id']}`",
        f"- UTC 実行開始: {manifest.get('started_at_utc', '未記録')} / 終了: {manifest.get('ended_at_utc') or '未記録'}",
        f"- 計画 {manifest.get('planned_requests', '未記録')} 回に対し、記録 {len(rows)} 回、成功 {states['ok']} 回。",
        f"- その他: 打切り {states['truncated']} 回 / 空応答 {states['empty']} 回 / ブロック {states['blocked']} 回 / エラー {states['error']} 回。",
        f"- 実行状態: `{manifest.get('status', '未記録')}`",
        f"- 基礎の『同じ質問を条件ごとに3回』: {'全計画条件で成功応答が3回以上ある。回答差の記述は本人が追記する。' if readiness['basic_three_repeats'] else '全計画条件では未充足。下表の件数を確認する。'}",
        "- 会話履歴の実装・動作確認の有無は、このベンチマーク記録からは判定していない。", "",
        "## 2. 比較条件", "",
        f"使用モデル: {', '.join('`' + m + '`' for m in manifest['models'])}", "",
        "| ID | 質問 |", "| --- | --- |",
    ]
    parts += [f"| {_md(p['id'])} | {_md(p['question'])} |" for p in manifest["prompts"]]
    parts += [
        "", f"- 反復計画: 各条件 {manifest.get('repeats', '未記録')} 回。",
        f"- 出力上限: {settings.get('max_output_tokens', '未記録')} tokens / temperature: `{settings.get('temperature', '未記録')}`。",
        f"- 最小送信間隔: {settings.get('min_interval_seconds', '未記録')} 秒 / タイムアウト: {settings.get('timeout_ms', '未記録')} ms。",
        f"- SDK の最大試行回数: {settings.get('sdk_attempts', '未記録')}。待機時間は `scheduled_wait_seconds` に別記。",
        "- 応答時間は API 呼出し開始から返却までの経過時間。通信・処理・生成を含み、純粋な生成速度や最初のトークンまでの時間ではない。",
        "- temperature はモデルの既定値を使う。モデル内部の設定まで同一であるとは扱わない。",
        "- トークン数は API が返した項目を保存し、未取得は空欄/nullのまま扱う。総数を入力と表示出力の和で代用しない。", "",
        "## 3. 実測結果", "",
        "中央値 [最小値, 最大値] を示す。括弧内はその項目の取得件数/成功件数。成功以外は分布に混ぜず、件数と本文を残す。", "",
        "| モデル | 質問 | 成功/全試行 | 応答時間 秒 | 入力 tokens | 出力 tokens |", "| --- | --- | ---: | --- | --- | --- |",
    ]
    parts += [f"| {_md(s['model'])} | {_md(s['question_id'])} | {s['ok']}/{s['attempts']} | {_stat_cell(s, 'latency_seconds')} | {_stat_cell(s, 'input_tokens')} | {_stat_cell(s, 'output_tokens')} |" for s in summaries]
    parts += ["", "### 各試行の応答時間", "", "| モデル | 質問 | 成功応答の全観測値（秒） |", "| --- | --- | --- |"]
    parts += [f"| {_md(s['model'])} | {_md(s['question_id'])} | {', '.join(_fmt(v) for v in s['latency_seconds_values']) or '未取得'} |" for s in summaries]
    parts += [
        "", "思考・総数・キャッシュのトークン数と各観測値は `summary.csv`、個別の完全な記録は `results.jsonl` にある。", "",
        "![応答時間の比較](latency.svg)", "",
        "## 4. 回答内容の確認", "",
        "`responses.md` の本文を確認し、`manual_quality_review.csv` に評価理由を記入する。", "",
        "- factual_agreement_0_1_2: 0=明確な誤り、1=一部正しい/不十分、2=参照情報と整合。参照情報が足りない場合は空欄として理由を記す。",
        "- unsupported_claims: 根拠を確認できない具体的主張を本文から記録する。",
        "- uncertainty_handling: 知らない情報をどう扱ったかを記録する。",
        "- format_compliance: 指定した形式や長さを満たすかを記録する。",
        "- 文字列の完全一致や文章量を、意味内容の正確さの代わりに使わない。", "",
        "### 自分で確かめたこと（提出前に記入）", "",
        "1. 一般知識の質問で、回答の内容・書き方がどう変わったか: 【記入】",
        "2. 講座固有の質問を用語集と照合した結果: 【記入】",
        "3. 応答時間やトークン数と、回答の有用性の関係: 【記入】",
        "4. 参照情報を添えた条件を実行した場合、その違い: 【未実行なら未実行と記入】", "",
        "## 5. 無料枠の制限との関係", "",
        "RPMは毎分リクエスト数、入力TPMは毎分入力トークン数、RPDは毎日リクエスト数。いずれかの制限に達すると利用が制限される。Googleはプロジェクト単位の適用と、モデル・利用枠による違いを案内している。", "",
        "実行時に設定された上限（AI Studioで自分のプロジェクトを確認した値だけを確定値として使う）:", "",
        "| モデル | RPM | 入力TPM | RPD | 値の出所 |", "| --- | ---: | ---: | ---: | --- |",
    ]
    for model in manifest["models"]:
        quota = settings.get("quotas", {}).get(model, {})
        parts += [f"| {_md(model)} | {_fmt(quota.get('rpm'))} | {_fmt(quota.get('tpm'))} | {_fmt(quota.get('rpd'))} | {_md(quota.get('source') or '未確認')} |"]
    n429 = sum(r.get("http_status") == 429 for r in rows)
    n503 = sum(r.get("http_status") == 503 for r in rows)
    wait = sum(r.get("scheduled_wait_seconds", 0) or 0 for r in rows)
    parts += [
        "", f"- 今回の記録: HTTP 429 は {n429} 回、HTTP 503 は {n503} 回。送信前待機は合計 {_fmt(wait)} 秒。",
        "- 429は利用枠関連、503はサービスの利用可能性に関する応答として区別する。429の原因をRPM・TPM・RPDのどれかに特定するには、AI Studioやエラー情報との照合が必要。",
        "- エラーがなかった場合も、上限そのものを測定できたとは言えない。故意に上限まで連続送信する実験は行わない。",
        "- 送信間隔を広げるとRPMへの余裕は増える。1日分の既利用回数、同じプロジェクトの他の利用、入力TPMの制限は別に確認する。",
        "- 回答が速いほど無制限に呼び出せるわけではない。履歴や参照文を追加すると入力トークン数が増える場合があり、本文の有用性と合わせて判断する。",
        "- 自分の上限と今回の実行数を照合した考察: 【AI Studioの確認日時・上限・残り回数などを記入】", "",
        "## 6. 再現性と結果の範囲", "",
        f"- Python: `{manifest.get('versions', {}).get('python', '未記録')}` / google-genai: `{manifest.get('versions', {}).get('google_genai', '未記録')}`",
        f"- 講座参照コミット: `{manifest.get('course_commit', '未記録')}`",
        "- 条件と時刻は `manifest.json`、呼出し順と返却モデルは `results.jsonl` に保存した。",
        "- 少数回の観測であり、サービスの混雑、通信、回答長、モデルの生成の揺れなどが影響する。モデルの恒常的な優劣を確定する結果ではない。",
        "- 小標本からp95や安定した信頼区間を示さず、全観測値と中央値・範囲を示した。",
        "- `live` の識別はテスト記録の取り違えを防ぐためのもの。記録自体の真正性を証明する電子署名ではない。", "",
        "記録ファイル: [実行条件](manifest.json) / [全試行](results.jsonl) / [集計CSV](summary.csv) / [回答本文](responses.md) / [本人による評価欄](manual_quality_review.csv)", "",
        "## 7. 提出フォームの記入", "",
    ]
    if achieved:
        q3 = (
            "複数モデルの共通質問に対し、応答時間と入力・出力トークン数を計測し、条件別の中央値・範囲と全応答を記録しました。"
            "工夫した点は、欠損値を0で埋めず、失敗や出力打切りも残し、429と503を区別したことです。"
            "無料枠のRPM・入力TPM・RPDと実験条件の関係を整理し、回答内容は用語集との照合欄で別に検証できる構成にしました。"
            "実測値まで追跡できる記録と再実行可能なコードを評価していただきたいです。"
        )
        parts += [
            "- Q1: 比較結果を読み、自分の考察を記入した上で「発展｜Advanced」を選ぶ。",
            "- Q2: 自分の公開forkのURLを記入する。公開とPRの作成はこのプログラムでは行っていない。",
            f"- Q3案（{len(q3)}文字。実際に確認した内容に合わせて修正）:", "", q3,
        ]
    else:
        parts += ["複数モデル×共通2質問の時間・入力/出力トークン測定が揃っていないため、発展を完了したとするQ3文案は生成しない。未完の条件と記録されたエラーを確認する。"]
    parts += [
        "", "## 8. 参考資料", "",
        "- [Google Gemini API: Rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)",
        "- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)",
        "- [講座リポジトリ](https://github.com/matsuolab/lecture-ai-engineering-2026)",
        "- 配布資料: 第1回演習講義資料 p.14〜16、宿題資料（配布PDFはこの公開提出物に含めない）。", "",
        "AI支援: コードと集計の下書きにChatGPTを利用。本人が実行・確認した範囲は上記の考察欄に記す。", "",
    ]
    return "\n".join(parts)


def generate_report(run_dir: Path, answer_path: Path | None = None) -> dict[str, str]:
    run_dir = Path(run_dir)
    manifest, rows = load_run(run_dir)
    answer_path = Path(answer_path) if answer_path is not None else run_dir / "answer.md"
    if answer_path.exists() and AUTO_MARKER not in answer_path.read_text(encoding="utf-8"):
        raise ValueError(f"既存の手書き解答を上書きしません: {answer_path}")
    summaries = summarize(manifest, rows)
    answer = build_answer(manifest, rows, summaries)
    paths = {
        "answer": answer_path, "summary": run_dir / "summary.csv", "responses": run_dir / "responses.md",
        "quality_review": run_dir / "manual_quality_review.csv", "chart": run_dir / "latency.svg",
    }
    # 手入力欄の形式検査を先に行い、その後に再生成可能な出力を書き込む。
    _quality_csv(paths["quality_review"], rows)
    _write_summary(paths["summary"], summaries)
    paths["responses"].write_text(_responses(manifest, rows), encoding="utf-8")
    paths["chart"].write_text(_chart(summaries), encoding="utf-8")
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    # 別の場所のanswer.mdに出す場合も、相対リンクが実データを指すようにする。
    if answer_path.parent.resolve() != run_dir.resolve():
        import os
        for filename in ("latency.svg", "manifest.json", "results.jsonl", "summary.csv", "responses.md", "manual_quality_review.csv"):
            relative = os.path.relpath(run_dir / filename, answer_path.parent).replace(os.sep, "/")
            answer = answer.replace("](" + filename + ")", "](" + relative + ")")
    answer_path.write_text(answer, encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Day1の実 API 記録から提出下書きを作る")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--answer", type=Path)
    args = parser.parse_args()
    try:
        paths = generate_report(args.run_dir, args.answer)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"集計できません: {exc}\n")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print("本人の観察・考察欄を記入してから提出してください。")


if __name__ == "__main__":
    main()
