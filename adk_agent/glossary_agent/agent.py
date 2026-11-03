"""Day3 発展(a): 演習と同じ指示・ツール・モデルを Google ADK で使う。"""

import os

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL

# 講座の Codespaces secret を使う。キーの値はコードや記録へ書き出さない。
# Pydantic AI と同じ API キー、同じ Gemini Developer API を使う。
if os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"

shared_tools = [tools.search_glossary, tools.get_lecture_summary]
# 演習で make_quiz を実装・登録済みの場合は、同じ関数も再利用する。
if callable(getattr(tools, "make_quiz", None)):
    shared_tools.append(tools.make_quiz)

root_agent = Agent(
    name="glossary_agent",
    model=MODEL,
    instruction=INSTRUCTIONS,
    tools=shared_tools,
)
