"""
test_intent_classification.py

Test client for the "Intent_classification" n8n workflow.

WHAT THE WORKFLOW DOES
-----------------------
The workflow exposes a webhook (node "Webhook", path
"dbce5b9e-1397-459a-871a-5b27433f1640"). It accepts a JSON payload describing
the latest customer message plus some conversation context, merges it,
sends it to a Gemini-powered "Classification Agent", and returns a
structured JSON object:

    {
      "route_to": "advisor" | "education" | "summary" | "unknown",
      "narrative": "<English summary of the conversation so far>"
    }

EXPECTED REQUEST PAYLOAD
-------------------------
    {
      "userMessage":   "<latest message from the customer>",
      "LastAImessage": "<the previous agent reply, if any>",
      "PrevAIagent":   "<name of the agent that produced LastAImessage>",
      "narrative":     "<running narrative carried over from prior turns>"
    }

These four fields match the "When Executed by Another Workflow" trigger's
input schema, and are also what the Webhook node forwards into the
Classification Agent via the "Merge input" node.

NOTE ON THE URL
-----------------
In n8n, a Webhook node has two URLs:
  - Test URL (used while you have "Listen for test event" open in the editor):
        {base_url}/webhook-test/{path}
  - Production URL (used once the workflow is Active, which this one is):
        {base_url}/webhook/{path}

Set N8N_BASE_URL below to your actual n8n instance (e.g. a self-hosted
domain or "http://localhost:5678" if running locally).
"""

import json
from dataclasses import dataclass, field

import requests

# ---------------------------------------------------------------------------
# Configuration — edit these for your environment
# ---------------------------------------------------------------------------
N8N_BASE_URL = "https://your-n8n-instance.com"  # <-- change me
WEBHOOK_PATH = "dbce5b9e-1397-459a-871a-5b27433f1640"
USE_TEST_URL = False  # True -> use the "Listen for test event" URL instead


def get_webhook_url() -> str:
    prefix = "webhook-test" if USE_TEST_URL else "webhook"
    return f"{N8N_BASE_URL}/{prefix}/{WEBHOOK_PATH}"


# ---------------------------------------------------------------------------
# Payload model
# ---------------------------------------------------------------------------
@dataclass
class IntentPayload:
    userMessage: str
    LastAImessage: str = ""
    PrevAIagent: str = ""
    narrative: str = ""

    def to_dict(self) -> dict:
        return {
            "userMessage": self.userMessage,
            "LastAImessage": self.LastAImessage,
            "PrevAIagent": self.PrevAIagent,
            "narrative": self.narrative,
        }


def call_intent_classifier(payload: IntentPayload, timeout: int = 30) -> dict:
    """POST a payload to the Intent Classification webhook and return the parsed JSON."""
    url = get_webhook_url()
    response = requests.post(url, json=payload.to_dict(), timeout=timeout)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Example payloads — one per routing category described in the system prompt
# ---------------------------------------------------------------------------
EXAMPLES = {
    # From the workflow's own pinData example: a pure question about a
    # program's eligibility -> should route to "education".
    "education": IntentPayload(
        userMessage="อยากรู้้ว่าปิดหนี้ไวไปต่อได้ คืออะไร",
        LastAImessage=(
            "โครงการ \"ปิดหนี้ไว ไปต่อได้\" เป็นมาตรการเฉพาะกิจที่ดำเนินการเพียงครั้งเดียว "
            "เพื่อช่วยเหลือลูกหนี้รายย่อยที่เป็นหนี้เสีย (NPLs) ที่มียอดหนี้ไม่สูง"
        ),
        PrevAIagent="education",
        narrative="",
    ),
    # Customer wants a personalized recommendation -> "advisor".
    "advisor": IntentPayload(
        userMessage="ผมมีหนี้บัตรเครดิตหลายใบ อยากได้แผนการชำระหนี้ที่เหมาะกับผมครับ",
        LastAImessage="",
        PrevAIagent="",
        narrative="",
    ),
    # Customer agrees to a plan the advisor already proposed -> "summary".
    "summary_after_advisor": IntentPayload(
        userMessage="ตกลงครับ ผมขอใช้แผนลดดอกเบี้ยตามที่แนะนำ",
        LastAImessage=(
            "จากรายได้และภาระหนี้ของคุณ แนะนำให้ปรับโครงสร้างหนี้ด้วยการลดอัตราดอกเบี้ย "
            "และขยายระยะเวลาผ่อนชำระ"
        ),
        PrevAIagent="advisor",
        narrative="Customer has multiple credit card debts and was given a restructuring "
        "recommendation (interest rate reduction) by the advisor.",
    ),
    # Off-topic message -> "unknown".
    "unknown_offtopic": IntentPayload(
        userMessage="วันนี้อากาศเป็นยังไงบ้าง",
        LastAImessage="",
        PrevAIagent="",
        narrative="",
    ),
}


def run_examples() -> None:
    for label, payload in EXAMPLES.items():
        print(f"\n=== {label} ===")
        print("Request: ", json.dumps(payload.to_dict(), ensure_ascii=False, indent=2))
        try:
            result = call_intent_classifier(payload)
            print("Response:", json.dumps(result, ensure_ascii=False, indent=2))
        except requests.exceptions.RequestException as exc:
            print(f"Request failed: {exc}")


if __name__ == "__main__":
    run_examples()
