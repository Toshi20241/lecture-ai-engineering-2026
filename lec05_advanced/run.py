#!/usr/bin/env python3
"""第5回発展：Treeの候補文を改良し、3方式を同条件で比較する。

使い方（リポジトリの直下から）:
    uv run python lec05_advanced/run.py --check-only
    uv run python lec05_advanced/run.py

5問×3方式×2回を順次実行。中断後も同じコマンドで再開できる。
回答の正しさは自動採点しない。実行ログと根拠を確認して採点する。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
VERSION = "1.0"
ORIGINAL_SHA256 = "721a57b79b7194a626da61c376ca12c16295dcb36492c1714d199b793afbdeca"
DOCUMENT_IDS = ("lec01-main", "lec02-main", "lec03-main")
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new_or_identical(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"既存の内容が異なるため上書きを中止しました: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def prepared_source(original: str) -> str:
    needle = next((line for line in original.splitlines()
                   if 'info["content"] = re.split' in line), None)
    if not needle or original.count(needle) != 1:
        raise RuntimeError("候補文の変更対象が一意に見つかりません。")
    anchor = "from pathlib import Path\n"
    if original.count(anchor) != 1:
        raise RuntimeError("importの挿入先が一意に見つかりません。")
    replacement = ('                info["content"] = tree_candidate_text('
                   'info["content"], info["title"], info.get("section", ""))')
    changed = original.replace(needle, replacement)
    changed = changed.replace(anchor, anchor + "from app.tree_preview import tree_candidate_text\n", 1)
    ast.parse(changed)
    return changed


def install() -> None:
    for name in ("app/retrieval.py", "app/tools.py", "app/agent.py",
                 "data/rag_data/lectures.json", "data/rag_data/tree.json"):
        if not (ROOT / name).is_file():
            raise RuntimeError(f"リポジトリ内で実行してください。ファイルがありません: {name}")
    original_bytes = (PACKAGE / "retrieval_original.py").read_bytes()
    if digest(original_bytes) != ORIGINAL_SHA256:
        raise RuntimeError("同梱の元コードが想定と異なります。配布ZIPを確認してください。")
    patched = prepared_source(original_bytes.decode("utf-8")).encode("utf-8")
    current_path = ROOT / "app/retrieval.py"
    current = current_path.read_bytes()
    if current not in (original_bytes, patched):
        raise RuntimeError("app/retrieval.py に別の変更があります。自動上書きを中止しました。")
    helper = (PACKAGE / "tree_preview.py").read_bytes()
    ast.parse(helper.decode("utf-8"))
    helper_path = ROOT / "app/tree_preview.py"
    if helper_path.exists() and helper_path.read_bytes() != helper:
        raise RuntimeError("app/tree_preview.py に別の内容があります。自動上書きを中止しました。")
    backup = ROOT / "lectures/05/advanced_backup/retrieval_before.py.txt"
    write_new_or_identical(backup, original_bytes)
    write_new_or_identical(helper_path, helper)
    if current != patched:
        temporary = current_path.with_suffix(".lec05_tmp")
        temporary.write_bytes(patched)
        os.replace(temporary, current_path)
    print("[準備] Tree候補文の改良を適用しました。", flush=True)


def load_helper():
    spec = importlib.util.spec_from_file_location("lec05_preview_check", ROOT / "app/tree_preview.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("候補文モジュールを読み込めません。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def offline_check() -> dict:
    """実データを使い、APIを呼ばずに改良の作用範囲を検証する。"""
    helper = load_helper()
    source = (ROOT / "app/retrieval.py").read_text(encoding="utf-8")
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name in ("load_pages", "browse_tree")]
    namespace = {
        "Path": Path, "json": json, "re": re,
        "RAG_SOURCE": ROOT / "data/rag_data/lectures.json",
        "RAG_TREE": ROOT / "data/rag_data/tree.json",
        "RAG_DOCUMENT_IDS": DOCUMENT_IDS,
        "tree_candidate_text": helper.tree_candidate_text,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), "retrieval_check", "exec"), namespace)
    pages = namespace["load_pages"]()
    if len(pages) != 214:
        raise RuntimeError(f"対象ページ数が配布版の214と異なります: {len(pages)}")
    browse = namespace["browse_tree"]
    tree = json.loads((ROOT / "data/rag_data/tree.json").read_text(encoding="utf-8"))
    tree["children"] = [node for node in tree["children"] if node["id"] in DOCUMENT_IDS]
    nodes = []
    stack = [tree]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.get("children", []))
    old_env = os.environ.get("AIE_TREE_PREVIEW")
    examples = []
    totals = {"original_preview_chars": 0, "improved_preview_chars": 0}
    try:
        for page in pages:
            os.environ["AIE_TREE_PREVIEW"] = "original"
            old = helper.tree_candidate_text(page["content"], page["title"], page["section"])
            expected = re.split(r"(?<=[。！？])|\n", page["content"].strip(), maxsplit=1)[0]
            if old != expected:
                raise AssertionError("配布版の再現が一致しません: " + page["page_id"])
            os.environ["AIE_TREE_PREVIEW"] = "improved"
            new = helper.tree_candidate_text(page["content"], page["title"], page["section"])
            if not new or len(new) > 200:
                raise AssertionError("候補文の長さが不正です: " + page["page_id"])
            totals["original_preview_chars"] += len(old)
            totals["improved_preview_chars"] += len(new)
            if page["page_id"] in ("lec03-main-p039", "lec03-main-p040", "lec03-main-p041", "lec02-main-p052"):
                examples.append({"page_id": page["page_id"], "title": page["title"],
                                 "original": old, "improved": new})
        # 一覧内の候補文以外に変化がないこと、ページ全文を切り詰めないことを検証。
        for node in nodes:
            os.environ["AIE_TREE_PREVIEW"] = "original"
            old = browse(node["id"])
            os.environ["AIE_TREE_PREVIEW"] = "improved"
            new = browse(node["id"])
            if old["node"] != new["node"]:
                raise AssertionError("ページ全文またはノード本体が変わっています: " + node["id"])
            if len(old["children"]) != len(new["children"]):
                raise AssertionError("階層が変わっています: " + node["id"])
            for a, b in zip(old["children"], new["children"]):
                a = dict(a)
                b = dict(b)
                if a["kind"] == "page":
                    a.pop("content", None)
                    b.pop("content", None)
                if a != b:
                    raise AssertionError("候補文以外の情報が変わっています: " + node["id"])
        # 一般の分数を進捗番号として削除しない。
        os.environ["AIE_TREE_PREVIEW"] = "improved"
        fraction = helper.tree_candidate_text("題名\n1/4 のメモリで動く\n本文です。", "題名")
        if "1/4 のメモリで動く" not in fraction:
            raise AssertionError("本文中の分数が失われました。")
        os.environ["AIE_TREE_PREVIEW"] = "invalid"
        try:
            helper.tree_candidate_text("本文", "題名")
        except ValueError:
            pass
        else:
            raise AssertionError("不正なモードが検出されませんでした。")
    finally:
        if old_env is None:
            os.environ.pop("AIE_TREE_PREVIEW", None)
        else:
            os.environ["AIE_TREE_PREVIEW"] = old_env
    result = {"version": VERSION, "status": "passed", "api_calls": 0,
              "pages_checked": len(pages), "tree_nodes_checked": len(nodes),
              "checks": ["配布版候補の完全再現", "改良候補が1〜200文字",
                         "全文・階層・出典情報の同一性", "本文の分数保持", "不正モード検知"],
              "character_counts_not_tokens": totals, "examples": examples}
    directory = ROOT / "lectures/05/advanced_design"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "offline_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[確認] {len(pages)}ページの候補文、{len(nodes)}ノードの全文・階層を検証しました（API呼び出し0回）。", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true", help="改良を適用し、APIなしの検証だけ実施")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="3方式で共通に用いるモデル名")
    parser.add_argument("--repeats", type=int, default=2, help="各問・各方式の成功試行数（既定2）")
    parser.add_argument("--questions", nargs="+", help="指定した問題IDのみ実行")
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 5:
        parser.error("--repeats は1〜5で指定してください。")
    try:
        cases = json.loads((PACKAGE / "cases.json").read_text(encoding="utf-8"))
        valid_ids = {case["id"] for case in cases}
        if args.questions and any(name not in valid_ids for name in args.questions):
            parser.error("--questions に不明なIDがあります。使用可能: " + ", ".join(sorted(valid_ids)))
        install()
        offline_check()
        design_dir = ROOT / "lectures/05/advanced_design"
        write_new_or_identical(design_dir / "cases.json", (PACKAGE / "cases.json").read_bytes())
        if args.check_only:
            print("[完了] APIなしの確認が完了しました。比較実行は --check-only を外して実行します。", flush=True)
            return 0
        from runner import run_comparison, make_bundle
        print("[比較] 基本RAG・配布版Tree・改良版Treeを同じ質問・モデル・資料で実行します。", flush=True)
        print("[比較] 正答・引用の評価は、実行後に本文と照合します。未評価の点数は作成しません。", flush=True)
        experiment_dir = run_comparison(ROOT, cases, model=args.model, repeats=args.repeats, selected_ids=args.questions)
        bundle = make_bundle(ROOT, experiment_dir)
        print(f"[結果] {bundle}", flush=True)
        print("このZIPをダウンロードしてチャットに添付してください。", flush=True)
        return 0
    except KeyboardInterrupt:
        print("\n[中断] 保存済みの結果を残しました。同じコマンドで再開してください。", flush=True)
        return 130
    except Exception as exc:
        print(f"[停止] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
