from base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from chatbot.utils.utils import get_api_key, DatabaseOps, EmailService
from typing import List, Dict, Optional, Union, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langchain.agents import create_tool_calling_agent, AgentExecutor


class EmailInput(BaseModel):
    """Input for the email sending tool."""
    subject: Optional[str] = Field(
        description="The subject of the email. Should be 'Users' Complaints'.",
        default="Users' Complaints"
    )
    body: str = Field(
        description="The body of the email containing the users' IDs, their complaints, and when they were submitted."
    )


class ComplaintHandlerAgent(BaseAgent):
    def __init__(self):
        self.db = DatabaseOps()
        self.email = EmailService()
        # self.llm = ChatCohere(
        #     cohere_api_key=get_api_key("COHERE"), model="command-r-plus-08-2024"
        # )
        self.llm = ChatOpenAI(api_key=get_api_key("OPENAI"), model="gpt-4o")

        super().__init__()

    def _init_prompt(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                {
                    "role": "system",
                    "content": """
                You are a helpful assistant for my RAG system. Your task is to analyze the chat 
                history and identify any user complaints. If you detect any complaints, send an 
                email with the subject line "Users' Complaints" summarizing the issues found. Do 
                not send an email if no complaints are present. The email must contain the user's
                ID, his complaint and when he submitted it.
                
                When you find complaints, organize them in a clear format like:
                
                User ID: [ID]
                Complaint: [Description of complaint]
                Time of Complaint: [Timestamp]
                
                Include multiple complaints if found, separated by blank lines.
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
                description="Retrieve the chat history and analyze it. This tool takes no input parameters.",
            ),
            Tool(
                name="send_email",
                func=self._send_email_wrapper,
                description="""
                Sends an email to the admin containing the users' complaints if any.
                Args:
                    body (str): Required. The body of the email containing the users' IDs, their complaints
                        and the time the complaints were submitted.
                    subject (str): Optional. The subject of the email. Defaults to "Users' Complaints".
                Example:
                    send_email(body="User ID: User 1\\nComplaint: Service is slow\\nTime: 2025-04-05")
                Returns:
                    Confirmation message that the email was sent successfully.
                """,
            ),
        ]

    def _init_agent(self):
        self.agent = create_tool_calling_agent(
            llm=self.llm, tools=self.tools, prompt=self.prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True
        )

    def analyze(self, query):
        result = self.agent_executor.invoke({"query": query})
        return result
    
    def _send_email_wrapper(self, body=None, subject=None, **kwargs):
        """
        Wrapper for send_email that handles different parameter combinations.
        """
        if isinstance(body, dict):
            email_body = body.get('body')
            email_subject = body.get('subject', "Users' Complaints")
            return self.send_email(subject=email_subject, body=email_body)
            
        if body:
            return self.send_email(subject=subject or "Users' Complaints", body=body)
            
        for k, v in kwargs.items():
            if isinstance(v, str) and len(v) > 20:
                return self.send_email(subject="Users' Complaints", body=v)
                
        return "Email was not sent: Missing required parameter 'body'"
        
    def send_email(self, subject: str = "Users' Complaints", body: Optional[str] = None) -> str:
        """
        Send an email to the admin with the users' complaints.
        
        Args:
            subject (str): The subject of the email.
            body (str): The body of the email containing the users' IDs, their complaints
                and the time the complaints were submitted.

        Returns:
            str: Confirmation message that the email was sent successfully.
        """
        if not body:
            return "Email was not sent: Body is required"
            
        self.email.receiver_email = "ali.mostafa.elneklawy@gmail.com"
        try:
            self.email.send_email(subject=subject, complaints=body)
        except Exception as e:
            return f"Email wasn't sent due to error: {e}"
        
        return f"Email was sent successfully with subject: '{subject}' and body:\n\n{body}"

    def get_chat_history(self, *args, **kwargs) -> str:
        """
        Retrieve and format chat history for the last 48 hours.

        Returns:
            str: Formatted chat history in markdown format
        """
        hist: List[Dict] = self.db.get_chat_history(
            full_history=True, last_n_hours=48
        )

        try:
            interactions: List = [user["interactions"] for user in hist]
        except TypeError as e:  # There were no interactions
            return "There were no conversations in the past 48 hours."

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
    agent = ComplaintHandlerAgent()

    result = agent.analyze("Are there any complaints?")
    print(result)
