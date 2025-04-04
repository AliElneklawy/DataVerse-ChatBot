import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from langchain.tools import Tool
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parents[1]
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from chatbot.utils.utils import get_api_key, DatabaseOps


class ChatHistortAnalyzerAgent:
    def __init__(self):
        self.db = DatabaseOps()
        self.llm = ChatCohere(
            cohere_api_key=get_api_key("COHERE"), model="command-r-plus-08-2024"
        )
        # self.llm = ChatOpenAI(api_key=get_api_key("OPENAI"), model="gpt-4o")
        self.prompt = self._init_prompt()
        self._init_tools()
        self._init_agent()

    def _init_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                {
                    "role": "system",
                    "content": """
                    You are a helpful assistant for my RAG system. Your job is to analyze 
                    the chat history and extract insights from it like the most common
                    5 questions, questions that weren't answered by the LLM, the time range at 
                    which we get the most messages, languages used, total number of users, common
                    patterns in questions and answers and other insights that you might find 
                    helpful. All this based on the user's query. Your analysis must be clear, 
                    insightful, and in-depth not consice. Don't provide the actual prompts but 
                    only provide insights. 
                    """,
                },
                {"role": "human", "content": "{query}"},
                {"role": "placeholder", "content": "{agent_scratchpad}"},
            ]
        )

        return prompt

    def _init_tools(self):
        self.tools = [
            Tool(
                name="chat_history_analyzer",
                func=self.get_chat_history,
                description="""
                Retrieve the chat history and analyze it.
                Args:
                    days (int): Number of days of history to analyze (e.g. 7 for one week)
                Returns:
                    Formatted chat history in markdown format
                """,
            )
        ]

    def _init_agent(self):
        self.agent = create_tool_calling_agent(
            llm=self.llm, tools=self.tools, prompt=self.prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True
        )

    def analyze(self, query):
        return self.agent_executor.invoke({"query": query})

    def get_chat_history(self, days: int = 7):
        """
        Retrieve and format chat history for the specified number of days.

        Args:
            days (int): Number of days of history to retrieve. Default is 7.

        Returns:
            Formatted chat history in markdown format
        """
        # print(days, type(days))
        hours = 24 * int(days)
        hist: List[Dict] = self.db.get_chat_history(
            full_history=True, last_n_hours=hours
        )

        try:
            interactions: List = [user["interactions"] for user in hist]
        except TypeError as e:  # There were no interactions
            return f"There were no conversations in the past {days} days."

        for user_interactions in interactions:
            for interaction in user_interactions:
                if "llm" in interaction:
                    del interaction["llm"]
                if "embedder" in interaction:
                    del interaction["embedder"]

        return self.format_chat_history_markdown(interactions)

    @staticmethod
    def format_chat_history_markdown(interactions_list: List) -> str:
        """Format the chat history into a markdown format for the LLM."""
        formatted_output = []

        formatted_output.append("# Chat History\n")

        for user_idx, user_interactions in enumerate(interactions_list, 1):
            formatted_output.append(f"## User {user_idx}\n")

            for interaction_idx, interaction in enumerate(user_interactions, 1):
                timestamp = interaction.get("timestamp", "Unknown time")
                user_msg = interaction.get("user", "")
                assistant_msg = interaction.get("assistant", "")

                try:
                    dt = datetime.fromisoformat(timestamp)
                    formatted_time = dt.strftime("%Y-%m-%d %H:%M")
                except (ValueError, TypeError):
                    formatted_time = timestamp

                formatted_output.append(
                    f"### Interaction {interaction_idx} ({formatted_time})\n"
                )
                formatted_output.append(f"**User:** {user_msg}\n")
                formatted_output.append(f"**Assistant:** {assistant_msg}\n")

            formatted_output.append("")

        return "\n".join(formatted_output)


if __name__ == "__main__":
    db = DatabaseOps()
    agent = ChatHistortAnalyzerAgent()

    result = agent.analyze("Show analysis for the chat history for the past 13 days")
    print(result)
