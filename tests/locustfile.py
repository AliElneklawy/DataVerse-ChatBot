from locust import HttpUser, task, between
import random


class WhatsAppBotUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def send_whatsapp_message(self):
        payload = {
            "Body": random.choice(
                [
                    "Hello, how are you?",
                    "What's the weather like?",
                    "Tell me a joke",
                    "Help me with something",
                ]
            ),
            "From": f"whatsapp:+{random.randint(1000000000, 9999999999)}",
            # "To": "whatsapp:+1234567890",  # Your Twilio number (optional, not strictly needed)
            "NumMedia": "0",
        }

        response = self.client.post(
            "/sms",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            print(f"Failed request: {response.status_code} - {response.text}")


if __name__ == "__main__":
    import os

    os.system("locust -f locustfile.py")
