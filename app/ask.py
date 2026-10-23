"""第1回: 1回質問と、履歴を保持する対話。

uv run python -m app.ask "LoRAとは何ですか？"
uv run python -m app.ask --chat

対話中は /reset で履歴を初期化、/exit または Ctrl-D で終了する。
APIキーは講座の app.config.get_api_key で環境変数から取得する。
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


class EmptyResponseError(RuntimeError):
    """本文を取得できなかった応答。履歴には追加しない。"""


def print_usage(response: Any) -> None:
    """APIが返した値を表示する。欠損値は実測ゼロと区別する。"""
    usage = getattr(response, "usage_metadata", None)
    names = (
        ("入力", "prompt_token_count"),
        ("出力", "candidates_token_count"),
        ("思考", "thoughts_token_count"),
        ("合計", "total_token_count"),
    )
    fields = []
    for label, name in names:
        value = getattr(usage, name, None)
        fields.append(f"{label}={value if value is not None else '未提供'}")
    print("[tokens] " + " / ".join(fields))


def visible_answer(response: Any) -> tuple[str, types.Content]:
    """未ブロックの候補を選び、思考partを除く。出力上限は明示する。"""
    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    block_reason = getattr(block_reason, "value", block_reason)
    if block_reason not in (None, "", "BLOCKED_REASON_UNSPECIFIED"):
        raise EmptyResponseError("要求がブロックされました。履歴は変更していません。")
    rejected_candidate = False
    for candidate in getattr(response, "candidates", None) or []:
        finish_reason = getattr(candidate, "finish_reason", None)
        finish_reason = getattr(finish_reason, "value", finish_reason)
        if finish_reason not in (None, "", "FINISH_REASON_UNSPECIFIED", "STOP", "MAX_TOKENS"):
            # SAFETY等の中断では、途中まで本文があっても採用しない。
            rejected_candidate = True
            continue
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        parts = content.parts or []
        text = "".join(
            part.text
            for part in parts
            if getattr(part, "text", None) and not getattr(part, "thought", False)
        )
        if text.strip():
            if finish_reason == "MAX_TOKENS":
                print("[warning] 出力上限に達しました。回答は途中までの内容です。")
            # SDKの返したpartを保ち、次のターンに必要な署名等も維持する。
            model_content = types.Content(role="model", parts=parts).model_copy(deep=True)
            return text, model_content
    if rejected_candidate:
        raise EmptyResponseError("回答生成がブロックまたは中断されました。履歴は変更していません。")
    raise EmptyResponseError("回答本文を取得できませんでした。履歴は変更していません。")


class Conversation:
    """成功したuser/modelの対だけを履歴に保持する。"""

    def __init__(self, client: Any, model: str = MODEL) -> None:
        self.client = client
        self.model = model
        self.history: list[types.Content] = []

    def reset(self) -> None:
        self.history.clear()

    def ask(self, question: str) -> str:
        if not question.strip():
            raise ValueError("質問を入力してください。")
        user = types.Content(role="user", parts=[types.Part.from_text(text=question)])
        pending = [*self.history, user]
        response = self.client.models.generate_content(model=self.model, contents=pending)
        print_usage(response)
        text, model_content = visible_answer(response)
        # 失敗・ブロック・空本文ならここに到達せず、前の履歴を維持する。
        self.history.extend((user, model_content))
        return text


def make_client() -> genai.Client:
    """暗黙の再試行を無効化し、1操作を1要求として扱う。"""
    return genai.Client(
        api_key=get_api_key(),
        http_options=types.HttpOptions(
            timeout=60_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def ask(question: str) -> str:
    """従来の1回質問のインターフェースを保つ。"""
    client = make_client()
    try:
        return Conversation(client).ask(question)
    finally:
        client.close()


def safe_error(exc: Exception) -> str:
    """例外の生メッセージや要求URLを出力しない。"""
    if isinstance(exc, EmptyResponseError):
        return str(exc)
    if isinstance(exc, errors.APIError):
        if exc.code == 429:
            return "APIの利用上限に達しました（429）。AI Studioで毎分・毎日の利用枠を確認してください。"
        return api_error_message(exc.code)
    return "通信またはSDKエラーが発生しました。接続・モデル名・実行環境を確認してください。"


def run_chat(client: Any, initial_question: str | None = None) -> None:
    session = Conversation(client)
    print("対話を開始します。/reset: 履歴を初期化、/exit: 終了")
    question = initial_question
    if question is not None:
        print("あなた> " + question)
    while True:
        if question is None:
            try:
                question = input("あなた> ").strip()
                if not sys.stdin.isatty():
                    # パイプ入力を保存したログにも、質問と回答の両方を残す。
                    print(question)
            except (EOFError, KeyboardInterrupt):
                print("\n対話を終了しました。")
                return
        if question in ("/exit", "/quit"):
            print("対話を終了しました。")
            return
        if question == "/reset":
            session.reset()
            print("履歴を初期化しました。")
        elif question:
            try:
                print("モデル> " + session.ask(question))
            except KeyboardInterrupt:
                print("\n対話を終了しました。")
                return
            except Exception as exc:
                print("[error] " + safe_error(exc))
        question = None


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMに質問する。--chatで会話履歴を保持")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保持して続けて質問")
    args = parser.parse_args()
    print(f"[model] {MODEL}")
    if args.chat:
        try:
            client = make_client()
        except Exception as exc:
            print("[error] " + safe_error(exc))
            raise SystemExit(1) from None
        try:
            run_chat(client, args.question)
        finally:
            client.close()
        return
    question = args.question if args.question is not None else DEFAULT_QUESTION
    print(f"[質問] {question}\n")
    try:
        print(ask(question))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        print("[error] " + safe_error(exc))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
