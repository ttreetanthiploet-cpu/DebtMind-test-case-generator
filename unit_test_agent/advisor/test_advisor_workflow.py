"""
test_advisor_workflow.py

Test client for the "AdvisorWorkFlow _v1.3" n8n workflow — UPDATED to match
the workflow's CURRENT webhook contract (confirmed against its pinned
example data), which is a FLATTENED payload, not the nested
sessionId/conversationDesc/userInfo/accInfo shape declared on the `Start`
sub-workflow trigger.

WHY FLATTENED?
----------------
The `Webhook` node feeds into `aggregate_webhook_input`:

    return { ...$input.first().json.body, inputPath: 'webhook' };

This just spreads the POST body's top-level keys straight through — it does
NOT replicate the flattening that the `parsing input for extractor` code
node performs for the `Start` (sub-workflow) path. But the `Debt Solution
Extractor` agent's prompt template reads flat fields directly:
`$json.accText`, `$json.consultAcc`, `$json.DebtSituation`,
`$json.maxPayment`, `$json.maxTerm`, `$json.refPlanID`,
`$json.maxPaymentY2`, `$json.maxPaymentY3`, `$json.offerText`,
`$json.narrative`. So whatever you POST to the webhook must already be in
that flat shape, or those prompt variables come through empty/undefined.

NOTE: this is arguably a gap in the workflow (the two entry points — `Start`
and `Webhook` — aren't equivalent), but until/unless that's fixed on the
n8n side, this script targets the contract the workflow actually expects
today.

WHAT THE WEBHOOK RETURNS
---------------------------
Only the raw extractor output:

    {
      "consultAcc": "...", "maxPayment": 0, "maxTerm": 0,
      "DebtSituation": "...", "refPlanID": null,
      "maxPaymentY2": null, "maxPaymentY3": null, "reConfirmMessage": ""
    }

(The offer-engine HTTP call and reply templates only run on the `Start`
sub-workflow path, not reachable from this webhook.)

REQUIRED PAYLOAD FIELDS
--------------------------
    {
      "consultAcc": "string",          # comma-separated account numbers, or ""
      "DebtSituation": "string|null",  # one of the 5 situation enums, or null
      "maxPayment": number|null,
      "maxTerm": number|null,
      "refPlanID": "string|null",
      "maxPaymentY2": number|null,
      "maxPaymentY3": number|null,
      "narrative": "string",
      "accText": "<JSON-stringified array of account summaries>",
      "offerText": "<JSON-stringified array of offered plans>",
      "accInfotoExtr": [ {account_number, port, creditlimit, os, installment, remaining_term}, ... ],
      "LastAImessage": "<JSON-stringified string — i.e. the bot's previous reply, double-quoted>",
      "userMessage": "string"
    }

This script provides a builder (`AdvisorWebhookPayload`) that takes natural
inputs (raw account records, a plain previous-bot-message string, an
offerSoln list) and produces this exact flattened shape for you — mirroring
`parsing input for extractor`'s logic in Python — so you don't have to
hand-build `accText`/`offerText`/`LastAImessage` yourself.

URL NOTE
----------
- Test URL  (n8n editor "Listen for test event" open): {base}/webhook-test/{path}
- Production URL (workflow Active, which this one is): {base}/webhook/{path}
"""

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration — edit these for your environment
# ---------------------------------------------------------------------------
N8N_BASE_URL = "https://your-n8n-instance.com"  # <-- change me
WEBHOOK_PATH = "b7607735-bac3-4965-8c68-2ed0ef38aec4"
USE_TEST_URL = False  # True -> use the "Listen for test event" URL instead


def get_webhook_url() -> str:
    prefix = "webhook-test" if USE_TEST_URL else "webhook"
    return f"{N8N_BASE_URL}/{prefix}/{WEBHOOK_PATH}"


# ---------------------------------------------------------------------------
# Shared account fixtures (taken from the workflow's own pinned example)
# ---------------------------------------------------------------------------
BASE_ACC_INFO: list[dict[str, Any]] = [
    {
        "accNo": "1xxxxxxx0974",
        "port": "สินเชื่อส่วนบุคคล",
        "creditLimit": 30000,
        "os": 19913.56,
        "installment": 1700,
        "remainTerm": 14,
    },
    {
        "accNo": "1xxxxxxx0975",
        "port": "สินเชื่อส่วนบุคคล",
        "creditLimit": 80000,
        "os": 72742.01,
        "installment": 3500,
        "remainTerm": 27,
    },
]


def transform_acc_info(acc_info: list[dict]) -> list[dict]:
    """Mirrors the `accUsed` mapping inside `parsing input for extractor`."""
    return [
        {
            "account_number": acc.get("accNo"),
            "port": acc.get("port"),
            "creditlimit": acc.get("creditLimit"),
            "os": acc.get("os"),
            "installment": acc.get("installment"),
            "remaining_term": acc.get("remainTerm"),
        }
        for acc in acc_info
    ]


def stringify_last_ai_message(content: str) -> str:
    """Mirrors `JSON.stringify($input.first().json.LastAImessage.content)`."""
    return json.dumps(content, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Payload model
# ---------------------------------------------------------------------------
@dataclass
class AdvisorWebhookPayload:
    userMessage: str
    consultAcc: str = ""
    DebtSituation: Optional[str] = None
    maxPayment: Optional[float] = None
    maxTerm: Optional[int] = None
    refPlanID: Optional[str] = None
    maxPaymentY2: Optional[float] = None
    maxPaymentY3: Optional[float] = None
    narrative: str = ""
    accInfo: list = field(default_factory=lambda: deepcopy(BASE_ACC_INFO))
    offerSoln: list = field(default_factory=list)  # e.g. [{"planId": "P1", "term": 36, "installment": 5000}]
    lastAIMessageContent: str = ""  # plain text of the bot's previous reply (or "" if none)

    def to_dict(self) -> dict:
        acc_used = transform_acc_info(self.accInfo)
        return {
            "consultAcc": self.consultAcc,
            "DebtSituation": self.DebtSituation,
            "maxPayment": self.maxPayment,
            "maxTerm": self.maxTerm,
            "refPlanID": self.refPlanID,
            "maxPaymentY2": self.maxPaymentY2,
            "maxPaymentY3": self.maxPaymentY3,
            "narrative": self.narrative,
            "accText": json.dumps(acc_used, ensure_ascii=False, separators=(",", ":")),
            "offerText": json.dumps(self.offerSoln, ensure_ascii=False, separators=(",", ":")),
            "accInfotoExtr": acc_used,
            "LastAImessage": stringify_last_ai_message(self.lastAIMessageContent),
            "userMessage": self.userMessage,
        }


def call_advisor(payload: AdvisorWebhookPayload, timeout: int = 60) -> dict:
    """POST a payload to the Advisor webhook and return the parsed JSON response."""
    url = get_webhook_url()
    response = requests.post(url, json=payload.to_dict(), timeout=timeout)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Example payloads
# ---------------------------------------------------------------------------

# 1) Baseline — reproduces the workflow's own pinned Webhook example exactly.
#    Two known accounts, customer asks for a new combined payment of 3,000.
BASELINE_LAST_AI_CONTENT = (
    "น้องฟิน ยินดีให้บริการคำปรึกษาแก้ปัญหาหนี้ค่ะ เพื่อให้ระบบสามารถวิเคราะห์แนวทางการแก้ปัญหาหนี้ได้ถูกต้อง "
    "กรุณาระบุความสามารถในการชำระสินเชื่อและระบุบัญชีที่ลูกค้าต้องการ จากบัญชีสินเชื่อธนาคารกรุงไทยที่พบบนระบบ ดังนี้ค่ะ\n\n"
    "บัญชีที่ 1\nเลขที่บัญชี: 1xxxxxxx0974\nประเภทสินเชื่อ: สินเชื่อส่วนบุคคล\nยอดชำระคงเหลือ: 19,913.56 บาท\n"
    "อัตราผ่อนชำระ: 1,700 บาท/เดือน\nระยะเวลาคงเหลือ: 14 เดือน\n\n"
    "บัญชีที่ 2\nเลขที่บัญชี: 1xxxxxxx0975\nประเภทสินเชื่อ: สินเชื่อส่วนบุคคล\nยอดชำระคงเหลือ: 72,742.01 บาท\n"
    "อัตราผ่อนชำระ: 3,500 บาท/เดือน\nระยะเวลาคงเหลือ: 27 เดือน"
)

EXAMPLES: dict[str, AdvisorWebhookPayload] = {
    "baseline_two_accounts_new_payment": AdvisorWebhookPayload(
        userMessage="ทั้งสองบัญชีนี้อยากผ่อนแค่เดือนละ 3000 ได้ไหม",
        consultAcc="",
        DebtSituation="DebtBurden",
        maxPayment=5200,
        maxTerm=360,
        narrative=(
            "The customer specifies their payment ability, stating they can only "
            "afford 3,000 THB per month for their two personal loan accounts "
            "combined. They are asking for a new repayment plan based on this "
            "amount. Routing to the advisor to create a personalized solution."
        ),
        lastAIMessageContent=BASELINE_LAST_AI_CONTENT,
    ),
    # 2) First contact — no prior state, customer describes a job-loss
    #    situation and asks to include ALL accounts at a lower payment over
    #    5 years -> EXTRACT mode.
    "first_contact_all_accounts": AdvisorWebhookPayload(
        userMessage=(
            "ผมเพิ่งตกงานครับ รายได้หายไปเลย อยากขอปรับลดยอดผ่อนทุกบัญชีให้เหลือ"
            "เดือนละ 2000 บาท ผ่อนให้จบภายใน 5 ปี"
        ),
        consultAcc="",
        DebtSituation=None,
        maxPayment=None,
        maxTerm=None,
        narrative="",
        lastAIMessageContent="",
    ),
    # 3) Follow-up that only updates the term — accounts, payment, and
    #    situation should be carried forward unchanged by the extractor.
    "update_term_only": AdvisorWebhookPayload(
        userMessage="อยากผ่อนให้จบภายใน 2 ปีครับ",
        consultAcc="1xxxxxxx0974,1xxxxxxx0975",
        DebtSituation="DebtBurden",
        maxPayment=3000,
        maxTerm=360,
        narrative="Customer already selected both accounts and a 3,000 THB/month budget.",
        lastAIMessageContent="รับทราบค่ะ กำลังประมวลผลแผนให้อยู่ค่ะ",
    ),
    # 4) CLARIFY mode — a plan has already been offered (offerSoln non-empty)
    #    and the customer reacts vaguely ("too expensive") with no concrete
    #    number -> reConfirmMessage should resolve to CLARIFY_HIGH_INSTALLMENT.
    "clarify_high_installment": AdvisorWebhookPayload(
        userMessage="แผนที่เสนอผ่อนแพงไปค่ะ ขอลดลงได้ไหม",
        consultAcc="1xxxxxxx0974,1xxxxxxx0975",
        DebtSituation="DebtBurden",
        maxPayment=5000,
        maxTerm=36,
        narrative="Advisor offered PLAN-001 (5,000 THB/month, 36 months); customer finds it too expensive.",
        offerSoln=[{"planId": "PLAN-001", "term": 36, "installment": 5000}],
        lastAIMessageContent="น้องฟินขอเสนอแผนหมายเลข PLAN-001 ผ่อนชำระ 5,000 บาทต่อเดือน เป็นเวลา 36 เดือนค่ะ",
    ),
}


def run_examples() -> None:
    for label, payload in EXAMPLES.items():
        print(f"\n=== {label} ===")
        print("Request: ", json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)[:800], "...")
        try:
            result = call_advisor(payload)
            print("Response:", json.dumps(result, ensure_ascii=False, indent=2))
        except requests.exceptions.RequestException as exc:
            print(f"Request failed: {exc}")


if __name__ == "__main__":
    run_examples()
