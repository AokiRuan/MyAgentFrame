import sys
import os

# Make sure MyAgentFrame root is on the path so ReAct/ and tool/ imports resolve
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ReAct"))

from llm_client import LLMClient
from tool.tool_executor import ToolExecutor
from tool.search_tool import search
from ReAct.react_agent import ReActAgent


def main():
    # 1. Init LLM client (reads DEEPSEEK_API_KEY / LLM_MODEL_ID / LLM_BASE_URL from .env)
    llm_client = LLMClient(timeout=60)

    # 2. Init tool executor and register search tool
    tool_executor = ToolExecutor()
    tool_executor.registerTool(
        name="search",
        description="Search the web for up-to-date information. Input: a search query string.",
        func=search,
    )

    # 3. Create ReAct agent
    agent = ReActAgent(llm_client=llm_client, tool_executor=tool_executor, max_steps=5)

    # 4. Run a test question
    question = "What is the latest version of Python and when was it released?"
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"{'='*50}\n")

    answer = agent.run(question)

    print(f"\n{'='*50}")
    print(f"Answer: {answer}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
