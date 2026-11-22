"""Treeの候補ページ用に、本文の短い抜粋を作る。

候補選択用の表示だけを変更する。ページ本文・ID・検索対象は変更しない。
外部APIや追加の依存パッケージは使わない。
"""

import os
import re
import unicodedata


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", "", text)


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def make_tree_preview(
    content: str,
    title: str = "",
    section: str = "",
    max_chars: int = 200,
) -> str:
    """題名の重複や資料のヘッダーを除き、冒頭の本文を最大200字で返す。

    文字数はPythonのlenによる文字数。トークン数ではない。
    本文中の数値・箇条書きは保持し、先頭の単独番号だけを省く。
    出典行の後3行以内に著作権行がある場合は、その出典ブロックも省く。
    全文を参照する必要は変わらない。
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    lines = [re.sub(r"[\t\u3000 ]+", " ", line).strip()
             for line in content.splitlines()]
    labels = {_normalized(value) for value in (title, section) if value.strip()}
    cleaned: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        norm = _normalized(line)
        if not line or norm in labels:
            i += 1
            continue
        # 分数に続く説明は本文として保持する。進捗だけ、または既知の見出し
        # が続く行に限って省く（例: 「4/4 選んだモデルを、いつ・どう直すか」）。
        fraction_prefix = re.fullmatch(r"\d+\s*/\s*\d+(?:\s+(.*))?", line)
        progress_line = fraction_prefix is not None and (
            not fraction_prefix.group(1)
            or _normalized(fraction_prefix.group(1)) in labels
        )
        if i < 4 and (
            progress_line
            or re.fullmatch(r"(?i:part|chapter)\s*\d+", line)
        ):
            i += 1
            continue
        if not cleaned and re.fullmatch(r"[\d①-⑳]+", line):
            i += 1
            continue
        if line.startswith(("©", "Ⓒ")):
            i += 1
            continue
        if re.match(r"^(?:出典|参考文献)\s*[：:）)]", line):
            # 第3回では出典が改行され、すぐ後の著作権行まで続く。
            copyright_at = next(
                (j for j in range(i + 1, min(i + 4, len(lines)))
                 if lines[j].startswith(("©", "Ⓒ"))),
                None,
            )
            i = copyright_at + 1 if copyright_at is not None else i + 1
            continue
        if re.match(r"(?i)^(?:arxiv:|https?://|doi:)", line):
            i += 1
            continue
        if norm.startswith("aiエンジニアリング") and re.fullmatch(
            r"aiエンジニアリング(?:実践)?講座(?:\d{4})?(?:第\d+回(?:[（(].*[）)])?)?",
            norm,
        ):
            i += 1
            continue
        if line == "【図表・画像内の補足文字起こし】":
            i += 1
            continue
        cleaned.append(line)
        # 同じ行が連続する抽出ノイズは表示を一つにする。
        if len(cleaned) >= 2 and _normalized(cleaned[-1]) == _normalized(cleaned[-2]):
            cleaned.pop()
        i += 1

    text = "\n".join(cleaned)
    if not text:
        text = title.strip()
    return _shorten(text, max_chars)


def tree_candidate_text(content: str, title: str, section: str = "") -> str:
    """環境変数で配布版と改良版の候補文を切り替える。

    AIE_TREE_PREVIEW=original: 配布版の先頭文/先頭行と完全に同じ処理。
    AIE_TREE_PREVIEW=improved: メタ情報を除いた本文抜粋（最大200字）。既定値。
    """
    mode = os.environ.get("AIE_TREE_PREVIEW", "improved")
    if mode == "original":
        return re.split(r"(?<=[。！？])|\n", content.strip(), maxsplit=1)[0]
    if mode == "improved":
        return make_tree_preview(content, title, section, max_chars=200)
    raise ValueError(
        "AIE_TREE_PREVIEWはoriginalまたはimprovedを指定してください: " + repr(mode)
    )
