import time
import json
import asyncio
import logging
import schedule
from threading import Thread
from datetime import datetime
from .utils import EmailService, DatabaseOps, create_folder

try:
    from .paths import CHAT_HIST_DIR
    from .inference import inference_pipeline
except ImportError:
    from paths import CHAT_HIST_DIR
    from inference import inference_pipeline

logger = logging.getLogger(__name__)


class BaseMonitorService:
    """
    Base class for monitoring services using the observer pattern.
    Manages email subscriptions and periodic task scheduling.
    """
    def __init__(self, 
                 email_service: EmailService, 
                 start_service: bool = True, 
                 every_hours: int = 24):
        """
        Initialize the BaseMonitorService with email service and scheduling options.

        Args:
            email_service (EmailService): The email service instance to use for notifications.
            start_service (bool): Whether to start the monitoring service immediately. Defaults to True.
            every_hours (int): Interval in hours for scheduled tasks. Defaults to 24.
        """
        self.db = DatabaseOps()
        self.email_service = email_service
        self.start_service = start_service
        self.every_hours = every_hours
        self._running = False
        self._thread = None

        logger.info(f"{self.__class__.__name__} initialized with _running = False")

        self.email_service.subscribe(self._on_email_change) # subscribe to the email service

        if self.start_service and self.email_service.receiver_email is not None:
            self._start_monitoring()

    def _on_email_change(self, old_email, new_email):
        """
        Callback method triggered when the email address changes.
        """
        logger.info(f"{self.__class__.__name__} Email changed from '{old_email}' to '{new_email}', running = {self._running}")
        if not old_email and new_email:
            self._start_monitoring()
        elif not new_email and old_email:
            # waiting for the background thread to join is blocking so we need it to run async
            asyncio.create_task(self._stop_monitoring())

    def _start_monitoring(self):
        """
        Start a separate thread to monitor responses periodically.
        Uses a scheduler to execute tasks at specified intervals.
        """
        if self.start_service and not self._running:
            schedule.every(self.every_hours).hours.do(self._retrieve_and_email)
            self._thread = Thread(target=self._run_scheduler, daemon=False)
            self._running = True
            self._thread.start()

            logger.info(f"{self.__class__.__name__} thread started. Will run every {self.every_hours} hours.")

    def _stop_monitoring(self):
        """Stop the monitoring thread and clean up resources."""
        if self._running:
            logger.info(f"{self.__class__.__name__}: Stopping monitoring, setting _running to False")
            schedule.clear()
            self._running = False
            if self._thread is not None:
                self._thread.join(timeout=5)  # Set a timeout to avoid hanging
                self._thread = None
            logger.info(f"{self.__class__.__name__}: thread stopped and cleaned up.")

    def _run_scheduler(self):
        logger.info(f"{self.__class__.__name__}: Scheduler loop starting with _running = True")
        while self._running:
            # logger.info(f"{self.__class__.__name__}: Scheduler checking for pending jobs...")
            schedule.run_pending()
            time.sleep(1)
        
        logger.info(f"{self.__class__.__name__}: Scheduler loop exited because _running = False")

    def _retrieve_and_email(self):
        """
        Abstract method to retrieve data and send email notifications.
        Must be implemented by subclasses.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")


class UncertainResponseMonitor(BaseMonitorService):
    """
    Monitors and reports uncertain responses from LLMs using a classifier.
    Inherits from BaseMonitorService.
    """
    def _retrieve_and_email(self):
        unknowns = []
        q_a = self.db.get_monitored_resp()
        logger.info(f"Fetched {len(q_a)} responses from the last 24 hours.")

        for question, answer in q_a:
            pred = inference_pipeline(answer[:60])
            if pred[0] == 1: # the LLM didn't know the answer
                unknowns.append((question, answer))  

        if unknowns:
            logger.info(f"Found {len(unknowns)} uncertain responses. Sending email...")
            subject = f"RAG System: {len(unknowns)} Uncertain Responses Detected"
            self.email_service.send_email(subject, unknowns)
        else:
            logger.info("No uncertain responses found.")


class ChatHistoryMonitor(BaseMonitorService):
    """
    Monitors and emails chat history periodically.
    Inherits from BaseMonitorService.
    """  
    def _retrieve_and_email(self):
        """
        Retrieve chat history, save it to a file, and send it via email.
        """
        now = datetime.now().strftime('%Y-%m-%d_%H:%M')
        receiver_email = self.email_service.receiver_email
        fname = f"{now}_{receiver_email}.json"

        history = self.db.get_chat_history(full_history=True)
        self._save_hist(history, fname)

        if history:
            logger.info(f"{self.__class__.__name__}: Found previous conversations. Sending email...")
            subject = "Chat History for Your Chatbot"
            self.email_service.send_email(subject, 
                                          json_data=history,
                                          filename=fname)
        else:
            logger.info(f"{self.__class__.__name__}: No chat history.")

    def _save_hist(self, hist, fname):
        """
        Save the chat history to a JSON file.

        Args:
            hist (list[dict]): The chat history data to save.
            fname (str): The filename for the saved history.
        """
        output_file = create_folder(CHAT_HIST_DIR) / fname
        with open(output_file, "w", encoding='utf-8') as outfile:
            json.dump(hist, outfile, indent=4, ensure_ascii=False)
