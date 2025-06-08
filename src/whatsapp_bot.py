import os
import logging
from twilio.rest import Client
from fastapi import FastAPI, Request
from fastapi.responses import Response
from chatbot.utils.utils import DatabaseOps
from chatbot.rag.cohere_rag import CohereRAG
from chatbot.utils.paths import INDEXES_DIR, WEB_CONTENT_DIR
from twilio.twiml.messaging_response import MessagingResponse

app = FastAPI()
logger = logging.getLogger(__name__)


class TwilioClient:
    TWILIO_SID = os.getenv("TWILIO_SID")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_NUMBER")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

    def __init__(self) -> None:
        self.client = Client(self.TWILIO_SID, self.TWILIO_AUTH_TOKEN)
        self.db = DatabaseOps()
        self.rag = CohereRAG(
            WEB_CONTENT_DIR / "bcaitech.txt", INDEXES_DIR, chunking_type="recursive"
        )

    def send_whatsapp_message(self, to_number, message):
        try:
            self.client.messages.create(
                from_=self.TWILIO_PHONE_NUMBER, body=message, to=f"whatsapp:{to_number}"
            )
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False

    async def webhook(self, request: Request):
        form_data = await request.form()
        incoming_msg = form_data.get("Body", "").lower()
        sender = form_data.get("From", "").replace("whatsapp:", "")

        existing_user = self.db.get_bot_sub(sender)
        if not existing_user:
            self.db.append_bot_sub(sender, first_name="Unknown", platform="WhatsApp")
            logger.info(f"New WhatsApp user {sender} added to the database.")

        resp = MessagingResponse()
        msg = resp.message()

        response = await self.rag.get_response(incoming_msg, sender)
        msg.body(response)

        return Response(content=str(resp), media_type="application/xml")


twilio_bot = TwilioClient()

app.post("/sms")(twilio_bot.webhook)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
