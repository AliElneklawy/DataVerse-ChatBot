import os
import json
import smtplib
import logging
import sqlite3
from pathlib import Path
from email import encoders
from typing import Optional, Any
from collections import defaultdict
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

try:
    from .paths import DATABASE_DIR
except ImportError:
    from paths import DATABASE_DIR


logger = logging.getLogger(__name__)


def create_folder(path: Path | str) -> Path:
    p = Path(path)
    if not p.exists():
        p.mkdir(exist_ok=True, parents=True)    
    return p

class DatabaseOps:
    def __init__(self):
        self.db_path = Path(create_folder(DATABASE_DIR)) / "history_and_usage.db"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                        user_id TEXT,
                        timestamp DATETIME,
                        question TEXT,
                        answer TEXT,
                        model_used TEXT,
                        embedding_model_used TEXT,
                        PRIMARY KEY (user_id, timestamp)
                        )
                """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_monitor (
                        user_id TEXT,
                        timestamp DATETIME,
                        model_used TEXT, 
                        embedding_model_used TEXT,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        request_cost INTEGER,
                        PRIMARY KEY (user_id, timestamp)
                        )
                """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bot_subscribers (
                        user_id INTEGER PRIMARY KEY,
                        first_name TEXT,
                        subscribed_on DATETIME,
                        platform TEXT
                        )
                """)
            
            conn.commit()

    def get_chat_history(self, 
                         user_id: Optional[str] = None, 
                         last_n: int = 3, 
                         full_history: bool = False,
                         last_n_hours: int = 24):
        with sqlite3.connect(self.db_path) as conn:
            if full_history:
                time_wndw = datetime.now() - timedelta(hours=last_n_hours)
                cursor = conn.execute("""
                    SELECT *
                    FROM chat_history 
                    WHERE  timestamp >= ?
                    ORDER BY user_id, timestamp
                """, (time_wndw,))
                results = cursor.fetchall()
                grouped_history = defaultdict(list)

                for uid, ts, question, answer, llm, embedder in results:
                    grouped_history[uid].append({
                        "timestamp": ts,
                        "user": question,
                        "assistant": answer,
                        "llm": llm,
                        "embedder": embedder
                    })

                history = [{"user_id": uid, "interactions": interactions} 
                            for uid, interactions in grouped_history.items()]
            else:
                cursor = conn.execute("""
                    SELECT question, answer
                    FROM chat_history
                    WHERE user_id = ?
                    ORDER BY timestamp
                    LIMIT ?
                    """, (user_id, last_n))
                history = [{"question": q, "answer": a} for q, a in cursor.fetchall()]

            return history if history else "No previous conversations."

    def append_chat_history(self, 
                            user_id: str, 
                            question: str, 
                            answer: str, 
                            model_used: str,
                            embedding_model_used: str) -> None:
        timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO chat_history (user_id, 
                                          timestamp, 
                                          question, 
                                          answer,
                                          model_used,
                                          embedding_model_used)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, timestamp, question, answer, model_used, embedding_model_used))
            conn.commit()
        
        logger.info(f"Appended chat history for user {user_id}")

    def append_cost(self, 
                    user_id: str, 
                    model_used: str,
                    embedding_model_used: str,
                    input_tokens: int,
                    output_tokens: int,
                    cost_per_input_token: float,
                    cost_per_output_token: float) -> None:
        timestamp = datetime.now()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO cost_monitor (user_id,
                                          timestamp,
                                          model_used,
                                          embedding_model_used,
                                          input_tokens,
                                          output_tokens, 
                                          request_cost)
                VALUES (?, ?, ?, ?, ?, ?, (? * ? + ? * ?))
                """, (user_id, timestamp, model_used, 
                      embedding_model_used, input_tokens, output_tokens,
                      input_tokens, cost_per_input_token/10**6,
                      output_tokens, cost_per_output_token/10**6,)) # price is per million tokens
            
            conn.commit()
        
        logger.info(f"Appended cost for user {user_id}")

    def get_monitored_resp(self) -> list[tuple[str, str]]:
        day_ago = datetime.now() - timedelta(hours=24) # get responses for the last 24 hours

        with sqlite3.connect(self.db_path) as conn:
            results = conn.execute(
                "SELECT question, answer FROM chat_history WHERE timestamp >= ?",
                (day_ago,)
            )
            q_a = results.fetchall()
        return q_a
    
    def append_bot_sub(self, user_id, first_name, platform):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO bot_subscribers (user_id,
                                            first_name,
                                            subscribed_on,
                                            platform
                                            )
                VALUES (?, ?, ?, ?)
            """, (user_id, first_name, datetime.now(), platform))

    def get_bot_sub(self, user_id=None):
        with sqlite3.connect(self.db_path) as conn:
            if user_id:
                results = conn.execute(f"""
                    SELECT user_id 
                    FROM bot_subscribers
                    WHERE user_id = ?
                """, (user_id,))
            else:
                results = conn.execute(f"""
                    SELECT *
                    FROM bot_subscribers
                """)
            return results.fetchall()


class EmailService:
    def __init__(self):
        self._subscribers = []
        self.sender_email = "bcaitech.ai@gmail.com"
        self._receiver_email = "melneklawy1966@gmail.com"
        self.app_password = os.getenv("GMAIL_APP_PASSWORD")

    def subscribe(self, callback):
        """Allow other classes to subscribe to email state changes."""
        if callable(callback) and callback not in self._subscribers:
            self._subscribers.append(callback)
    
    def unsubscibe(self, callback):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def _notify_subscribers(self, old_email, new_email):
        for callback in self._subscribers:
            try:
                callback(old_email, new_email)
            except Exception as e:
                logger.error(f"Error notifying subscibers: {e}")
        
        logger.info("Subscribers notified.")
    
    def _format_email_content(self, unknowns):
        """
        Format the email content with a table of uncertain responses.

        Args:
            unknowns (list[tuple[str, str]]): List of (question, answer) tuples.

        Returns:
            str: Formatted HTML content for the email.
        """
        email_content = "<h2>Uncertain Responses in RAG System</h2>"
        email_content += "<p>The following questions received uncertain responses and might need to be added to the knowledge base:</p>"
        email_content += "<table border='1'><tr><th>ID</th><th>Question</th><th>Response</th></tr>"

        for i, (question, answer) in enumerate(unknowns, 1):
            truncated_answer = (answer[:100] + "...") if len(answer) > 100 else answer
            email_content += f"<tr><td>{i}</td><td>{question}</td><td>{truncated_answer}</td></tr>"

        email_content += "</table>"
        return email_content

    def _send_without_attachment(self, message: MIMEMultipart, unknowns: list[tuple[str, str]]):
        """
        Prepare message without attachments for uncertain responses.

        Args:
            message (MIMEMultipart): The email message object
            unknowns (list[tuple[str, str]]): List of (question, answer) tuples

        Returns:
            MIMEText: HTML content for the message
        """
        html_content = self._format_email_content(unknowns)
        html_part = MIMEText(html_content, "html")
        return html_part

    def _add_file_attachment(self, message: MIMEMultipart, file_path: str, content_type=None):
        """
        Add a file attachment to the email message.

        Args:
            message (MIMEMultipart): The email message object
            file_path (str): Path to the file to attach
            content_type (str, optional): Content type of the file. Defaults to None.

        Returns:
            bool: True if attachment was successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return False
                
            filename = os.path.basename(file_path)
            
            with open(file_path, "rb") as attachment_file:
                attachment_data = attachment_file.read()
            
            # Determine content type if not specified
            if content_type is None:
                if file_path.lower().endswith('.json'):
                    content_type = 'application/json'
                elif file_path.lower().endswith('.xlsx'):
                    content_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                elif file_path.lower().endswith('.pdf'):
                    content_type = 'application/pdf'
                else:
                    content_type = 'application/octet-stream'
            
            attachment = MIMEBase(*content_type.split('/', 1))
            attachment.set_payload(attachment_data)
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', 'attachment', filename=filename)
            message.attach(attachment)
            
            return True
            
        except Exception as e:
            logger.error(f"Error attaching file {file_path}: {e}")
            return False

    def send_email_with_attachments(self, subject: str, message_body: str, file_paths: list[str] = None):
        """
        Send an email with multiple file attachments.

        Args:
            subject (str): The email subject
            message_body (str): The email body text
            file_paths (list[str], optional): List of file paths to attach. Defaults to None.
        """
        if not self._receiver_email:
            logger.warning("Receiver email was not set. Email wasn't sent.")
            return

        message = MIMEMultipart()
        message["Subject"] = subject
        message["From"] = self.sender_email
        message["To"] = self._receiver_email

        text_part = MIMEText(message_body, "plain")
        message.attach(text_part)

        if file_paths:
            for file_path in file_paths:
                self._add_file_attachment(message, file_path)

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.sender_email, self.app_password)
                server.sendmail(self.sender_email, self._receiver_email, message.as_string())
            logger.info(f"Email sent to {self._receiver_email} with subject: {subject} and {len(file_paths) if file_paths else 0} attachments")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def _send_with_attachment(self, message: MIMEMultipart, json_data: list[dict[str, Any]], filename: str):
        """
        Add JSON data as an attachment to the email.

        Args:
            message (MIMEMultipart): The email message object
            json_data (list[dict]): JSON data to attach
            filename (str): Filename for the attachment

        Returns:
            MIMEApplication: The JSON attachment
        """
        text_part = MIMEText("Please find the attached JSON file of your conversations.\n\nBest Regards,\n", "plain")
        message.attach(text_part)
        json_str = json.dumps(json_data, indent=4, ensure_ascii=False)
        attachment = MIMEApplication(json_str.encode('utf-8'))
        attachment.add_header('Content-Disposition', 'attachment', filename=filename)
        return attachment

    def send_email(self, 
                   subject: str, 
                   unknowns: list[tuple[str, str]] = None, 
                   json_data: list[dict[str, Any]] = None,
                   filename: str = "conversations.json"):
        """
        Send an email with either uncertain responses or JSON data.
        
        This method is maintained for backwards compatibility.

        Args:
            subject (str): The email subject line
            unknowns (list[tuple[str, str]], optional): List of uncertain responses. Defaults to None.
            json_data (list[dict], optional): JSON data to attach. Defaults to None.
            filename (str, optional): Filename for JSON attachment. Defaults to "conversations.json".
        """
        if not self._receiver_email:
            logger.warning("Receiver email was not set. Email wasn't sent.")
            return

        msg_type = "mixed" if json_data else "alternative"

        message = MIMEMultipart(msg_type)
        message["Subject"] = subject
        message["From"] = self.sender_email
        message["To"] = self._receiver_email

        if json_data:
            message.attach(self._send_with_attachment(message, json_data, filename))
        else:
            message.attach(self._send_without_attachment(message, unknowns))

        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(self.sender_email, self.app_password)
                server.sendmail(self.sender_email, self._receiver_email, message.as_string())
            logger.info(f"Email sent to {self._receiver_email} with subject: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    @property
    def receiver_email(self):
        return self._receiver_email
    
    @receiver_email.setter
    def receiver_email(self, value):
        old_value = self._receiver_email
        self._receiver_email = value
        logger.info(f"Email is set to {self._receiver_email}.")

        self._notify_subscribers(old_value, value)
