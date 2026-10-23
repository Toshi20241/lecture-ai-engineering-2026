"""APIを呼ばず、履歴・失敗時の状態・SDK応答処理を検証する。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from google.genai import types

from app import ask as ask_module


def response(text="回答", usage=True, thought=None, finish_reason=None, block_reason=None):
    parts = []
    if thought:
        parts.append(types.Part(text=thought, thought=True))
    if text is not None:
        parts.append(types.Part.from_text(text=text))
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=parts), finish_reason=finish_reason
        )],
        prompt_feedback=(
            types.GenerateContentResponsePromptFeedback(block_reason=block_reason)
            if block_reason else None
        ),
        usage_metadata=(
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=8, candidates_token_count=3, total_token_count=11
            )
            if usage else None
        ),
    )


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.closed = False
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        self.closed = True


def texts(contents):
    return [(c.role, "".join(p.text or "" for p in c.parts)) for c in contents]


def test_second_turn_includes_the_first_exchange_and_sessions_are_isolated():
    client = FakeClient([response("承知しました"), response("青です"), response("別の回答")])
    chat = ask_module.Conversation(client, model="test-model")
    assert chat.ask("合言葉は青です") == "承知しました"
    assert chat.ask("合言葉は？") == "青です"
    assert texts(client.calls[1]["contents"]) == [
        ("user", "合言葉は青です"),
        ("model", "承知しました"),
        ("user", "合言葉は？"),
    ]
    assert client.calls[1]["model"] == "test-model"
    other = ask_module.Conversation(client)
    other.ask("別セッション")
    assert texts(client.calls[2]["contents"]) == [("user", "別セッション")]


def test_reset_removes_previous_history():
    client = FakeClient([response("最初"), response("新規")])
    chat = ask_module.Conversation(client)
    chat.ask("質問1")
    chat.reset()
    chat.ask("質問2")
    assert texts(client.calls[1]["contents"]) == [("user", "質問2")]


@pytest.mark.parametrize("bad", [RuntimeError("do-not-log-me"), response(None)])
def test_failed_or_empty_response_does_not_commit_history(bad):
    client = FakeClient([response("成功"), bad, response("再開")])
    chat = ask_module.Conversation(client)
    chat.ask("質問1")
    before = deepcopy(chat.history)
    with pytest.raises((RuntimeError, ask_module.EmptyResponseError)):
        chat.ask("失敗する質問")
    assert chat.history == before
    chat.ask("質問3")
    assert texts(client.calls[2]["contents"]) == [
        ("user", "質問1"), ("model", "成功"), ("user", "質問3")
    ]


def test_missing_usage_is_unknown_and_hidden_thought_is_not_printed(capsys):
    client = FakeClient([response("表示する回答", usage=False, thought="非表示の思考")])
    chat = ask_module.Conversation(client)
    assert chat.ask("質問") == "表示する回答"
    output = capsys.readouterr().out
    assert "入力=未提供" in output and "出力=未提供" in output
    assert "非表示の思考" not in output
    assert chat.history[1].parts[0].thought is True


@pytest.mark.parametrize("blocked", [
    response("採用しない途中の本文", finish_reason="SAFETY"),
    response("採用しない途中の本文", finish_reason="RECITATION"),
    response("採用しない途中の本文", block_reason="SAFETY"),
])
def test_blocked_response_with_text_is_not_returned_or_committed(blocked, capsys):
    client = FakeClient([response("成功"), blocked])
    chat = ask_module.Conversation(client)
    chat.ask("最初の質問")
    before = deepcopy(chat.history)
    with pytest.raises(ask_module.EmptyResponseError, match="ブロック"):
        chat.ask("次の質問")
    assert chat.history == before
    assert "採用しない途中の本文" not in capsys.readouterr().out


def test_max_tokens_retains_partial_text_and_warns(capsys):
    client = FakeClient([response("途中の回答", finish_reason="MAX_TOKENS")])
    chat = ask_module.Conversation(client)
    assert chat.ask("質問") == "途中の回答"
    assert len(chat.history) == 2
    assert "回答は途中まで" in capsys.readouterr().out


def test_one_shot_preserves_string_return_and_closes_client(monkeypatch):
    client = FakeClient([response("回答")])
    monkeypatch.setattr(ask_module, "make_client", lambda: client)
    assert ask_module.ask("質問") == "回答"
    assert client.closed


def test_one_shot_closes_client_even_on_error(monkeypatch):
    client = FakeClient([RuntimeError("private-error-text")])
    monkeypatch.setattr(ask_module, "make_client", lambda: client)
    with pytest.raises(RuntimeError):
        ask_module.ask("質問")
    assert client.closed
    assert "private-error-text" not in ask_module.safe_error(RuntimeError("private-error-text"))


def test_interactive_reset_and_exit(monkeypatch, capsys):
    client = FakeClient([response("初回"), response("初期化後")])
    commands = iter(["質問1", "/reset", "質問2", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    ask_module.run_chat(client)
    assert texts(client.calls[1]["contents"]) == [("user", "質問2")]
    assert "対話を終了しました" in capsys.readouterr().out


def test_eof_exits_without_request(monkeypatch):
    client = FakeClient([])
    def eof(_):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    ask_module.run_chat(client)
    assert not client.calls
