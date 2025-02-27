import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

def send_slack_alert(title: str, image_url: str, captured_at: str):
    """
    Sends an alert to Slack with the errored frame details.

    :param title: The title of the alert
    :param image_url: Publicly accessible URL of the errored frame
    :param captured_at: Timestamp when the frame was captured
    """
    if not SLACK_WEBHOOK_URL:
        print("Slack Webhook URL is missing. Set SLACK_WEBHOOK_URL in .env file.")
        return

    payload = {
        "attachments": [
            {
                "fallback": title,
                "color": "#ff0000",  # Red color for error messages
                "title": title,
                "text": f"🚨 **Errored Frame Detected!**\n🕒 Captured at: {captured_at}\n🔴 This frame has an error.",
                "image_url": image_url if image_url else "https://media-hosting.imagekit.io//234ed92c190042c9/WhatsApp%20Image%202025-02-27%20at%2012.42.18.jpeg?Expires=1835270508&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=a30c0mvQwzqrzJIjWeoMkJstfIvZPtfWiCr8A978Jxyn~Zk8MeKOuApa-l007vD1Nm87lM-zSn9s9-g5-qFr-wIwqDxeDZZXxjtrs9MmACtnxEx1auN9TAnmy~RQoIXtf8NG3UTC~FlAJWqClowN0ZMo4i-93royHHNYewpY3Kbs5~7AIFMO0yHzm5lQ8MjT9mWP-0d-j5gQFzt6thLCC6lxupkLJsadm8qPtHyYHA2~JcS7ceOwEJqRaK2A-wnCSvK89WxVVWLMs2gRA6IhnhLyQ286l~J3DO-plapSfKR01Uz3I~a29PaA5K1ZX55Ck10ARKqgja4rreEVVKbP8w__",
                "footer": "Alert Service",
            }
        ]
    }

    response = requests.post(SLACK_WEBHOOK_URL, json=payload)
    
    if response.status_code == 200:
        print("✅ Alert sent to Slack successfully!")
    else:
        print(f"❌ Failed to send alert. Status Code: {response.status_code}, Response: {response.text}")


#call like this 
# Test Data
# title = "Frame Processing Error"
# image_url = None  # Replace with an actual public image URL
# captured_at = "2025-02-27 15:00:00"
# send_slack_alert(title, image_url, captured_at)

