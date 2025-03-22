import os
import json
import smtplib
import logging
import sqlite3
from pathlib import Path
from typing import Optional, Any
from collections import defaultdict
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
                         full_history: bool = False):
        with sqlite3.connect(self.db_path) as conn:
            if full_history:
                cursor = conn.execute("""
                    SELECT *
                    FROM chat_history 
                    ORDER BY user_id, timestamp
                """)
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
        day_ago = datetime.now() - timedelta(hours=21) # get responses for the last 24 hours

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

    def _send_without_attachment(self, 
                                 message: MIMEMultipart, 
                                 unknowns: list[tuple[str, str]]):
        html_content = self._format_email_content(unknowns)
        html_part = MIMEText(html_content, "html")

        return html_part

    def _send_with_attachment(self, 
                              message: MIMEMultipart, 
                              json_data: list[dict[str, Any]], 
                              filename: str):
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
                   filename: str = f"conversations.json"):
        """
        Send an email with the list of uncertain responses.

        Args:
            receiver_email (str): The recipient's email address.
            subject (str): The email subject line.
            unknowns (list[tuple[str, str]]): List of (question, answer) tuples to include in the email.
        """
        if not self._receiver_email:
            logger.warning("Reciver email was not set. Email wasn't sent.")
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




# class ResponseMonitor: # observer pattern
#     def __init__(self, email_service: EmailService):
#         self.db = DatabaseOps()
#         self.email_service = email_service
#         self._running = False
#         self._thread = None
#         logger.info("ResponseMonitor initialized with _running = False")

#         self.email_service.subscribe(self._on_email_change) # subscribe to the email service

#         if self.email_service.receiver_email is not None:
#             self._start_monitoring()

#     def _on_email_change(self, old_email, new_email):
#         logger.info(f"Email changed from '{old_email}' to '{new_email}', running = {self._running}")
#         if not old_email and new_email:
#             self._start_monitoring()
#         elif not new_email and old_email:
#             # waiting for the background thread to join is blocking so we need it to run async
#             asyncio.create_task(self._stop_monitoring())

#     def _start_monitoring(self):
#         """
#         Run a separate thread to monitor LLMs' responses.
#         Uses random forest classifier to classify LLMs' responses.
#         """
#         if not self._running:
#             schedule.every(30).seconds.do(self._retrieve_and_email)
#             self._thread = Thread(target=self._run_scheduler, daemon=False)
#             self._running = True
#             self._thread.start()

#             logger.info("Response monitoring thread started...")

#     async def _stop_monitoring(self):
#         if self._running:
#             logger.info("Stopping monitoring, setting _running to False")
#             schedule.clear()
#             self._running = False
#             if self._thread is not None:
#                 loop = asyncio.get_event_loop()
#                 await loop.run_in_executor(None, self._thread.join)
#                 # self._thread.join() # Wait for the thread to finish
#                 self._thread = None

#             logger.info("Response monitoring thread stopped and cleaned up.")

#     def _run_scheduler(self):
#         logger.info("Scheduler loop starting with _running = True")
#         while self._running: #===========
#             logger.info("Scheduler checking for pending jobs...")
#             schedule.run_pending()
#             time.sleep(30) # check every 30 seconds for pending jobs
        
#         logger.info("Scheduler loop exited because _running = False")

#     def _retrieve_and_email(self):
#         unknowns = []
#         q_a = self.db.get_monitored_resp()
#         logger.info(f"Fetched {len(q_a)} responses from the last 24 hours.")

#         for question, answer in q_a:
#             pred = inference_pipeline(answer[:60])
#             if pred[0] == 1: # the LLM didn't know the answer
#                 unknowns.append((question, answer))  

#         if unknowns:
#             logger.info(f"Found {len(unknowns)} uncertain responses. Sending email...")
#             subject = f"RAG System: {len(unknowns)} Uncertain Responses Detected"
#             self.email_service.send_email(subject, unknowns)
#         else:
#             logger.info("No uncertain responses found.")

#         # print(unknowns)

# class ChatHistorySender(ResponseMonitor):
#     def _start_monitoring(self):
#         """
#         Override to schedule ChatHistorySender's _retrieve_and_email.
#         """
#         if not self._running:
#             schedule.every(30).seconds.do(self._retrieve_and_email)
#             self._thread = Thread(target=self._run_scheduler, daemon=False)
#             self._running = True
#             self._thread.start()
#             logger.info("ChatHistorySender monitoring thread started...")

#     def _retrieve_and_email(self):
#         receiver_email = self.email_service.receiver_email
#         history = self.db.get_chat_history(full_history=True)
#         self.save_hist(history, receiver_email)

#         if history:
#             logger.info(f"Found previous conversations. Sending email...")
#             subject = "Chat History for Your Chatbot"
#             self.email_service.send_email(subject, 
#                                           json_data=history,
#                                           filename=f"conversations_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.json")
#         else:
#             logger.info("No chat history.")

#     def save_hist(self, hist, fname):
#         output_file = create_folder(CHAT_HIST_DIR) / f"{fname}.json"
#         with open(output_file, "w", encoding='utf-8') as outfile:
#             json.dump(hist, outfile, indent=4, ensure_ascii=False)