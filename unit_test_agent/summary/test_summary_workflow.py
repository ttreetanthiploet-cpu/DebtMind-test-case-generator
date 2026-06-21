"""
test_summary_workflow.py

Test client for the "Summary Workflow" n8n workflow.

WHAT THIS WORKFLOW DOES
--------------------------
Takes a finished debt-consultation conversation (customer profile, the
offer the customer selected — or didn't — and message history) and uses a
Gemini agent ("Summary Agent") to produce a structured Thai staff memo:

    {
      "subject": "...", "objective": "...", "debt_cause": "...",
      "offer_suitability": "...", "request_information": "...", "summary": "..."
    }

On the full sub-workflow path (`Start` trigger), this memo also gets
rendered into an HTML report and uploaded to Supabase Storage. That part is
NOT reachable from the public webhook (see below).

ENTRY POINTS — SAME PATTERN AS THE OTHER WORKFLOWS IN THIS PROJECT
-----------------------------------------------------------------------
- `Start` (executeWorkflowTrigger): only reachable as a sub-workflow. Runs
  the FULL preprocessing chain — `parsing history` -> `parsing last message
  before summary` -> `parsing offer to text` -> `summarise account` ->
  `parsing userInfo to text` -> `prepare_input_to_summarise` — which
  flattens the rich nested input (conversationDesc/userInfo/accInfo/
  selectedOffer/history) into the flat fields the `Summary Agent`'s prompt
  actually reads.
- `Webhook` (path "515736a7-e9eb-4ea0-81cb-bfd4a4248a8b"): reachable over
  HTTP. Its `aggregate_webhook_input` code is just
  `return {...$input.first().json.body, inputPath: 'webhook'}` — it does
  NOT run that preprocessing chain. So, exactly like the AdvisorWorkFlow,
  whatever you POST here must ALREADY be in the flattened shape, confirmed
  against this workflow's own pinned `Webhook` example (verified
  field-for-field to match what `prepare_input_to_summarise` produces).

WHAT THE WEBHOOK RETURNS
---------------------------
`parsing_output_to_webhook` does `return $input.first().json.output`, i.e.
just the Summary Agent's structured JSON output directly:

    {
      "subject": "...", "objective": "...", "debt_cause": "...",
      "offer_suitability": "...", "request_information": "...", "summary": "..."
    }

REQUIRED PAYLOAD FIELDS (flattened contract)
------------------------------------------------
    {
      "userMessage": "string",            # see note below
      "LastAImessage": "string",          # plain text, NOT JSON-stringified
      "userMessageSummary": "string",     # "message 1: ...\nmessage 2: ..."
      "narrative": "string",
      "preference": "string",
      "maxPayment": "string",             # "Not provided" or "<amount> THB"
      "ageRange": "string",               # e.g. "50-54 years old"
      "employmentType": "string",
      "monthsAsCustomer": number,
      "dpd": number,
      "sumOsNCB": number,
      "installmentNCB_Y1": number,
      "installmentNCB_Y2": number,
      "installmentNCB_Y3": number,
      "offerReadableText": "string"       # see note below
    }

NOTE on `userMessage`: if the customer selected an offer (`selectedOffer`
non-empty), the real workflow replaces userMessage with the literal string
"The user select one of the offer provided by AI" (see `parsing last
message before summary`). The builder below replicates that.

NOTE on `offerReadableText`: this is a free-text Thai rendering of the
selected offer, built by `parsing offer to text`. The builder below ports
that function line-for-line in Python so you can pass a structured offer
dict instead of hand-writing the formatted text. If no offer was selected,
the correct value is the literal string
"The user does not select the offer suggested by AI".

URL NOTE
----------
- Test URL  (n8n editor "Listen for test event" open): {base}/webhook-test/{path}
- Production URL (workflow Active, which this one is): {base}/webhook/{path}
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration — edit these for your environment
# ---------------------------------------------------------------------------
N8N_BASE_URL = "https://your-n8n-instance.com"  # <-- change me
WEBHOOK_PATH = "515736a7-e9eb-4ea0-81cb-bfd4a4248a8b"
USE_TEST_URL = False  # True -> use the "Listen for test event" URL instead


def get_webhook_url() -> str:
    prefix = "webhook-test" if USE_TEST_URL else "webhook"
    return f"{N8N_BASE_URL}/{prefix}/{WEBHOOK_PATH}"


# ---------------------------------------------------------------------------
# Ported transformation logic (mirrors the workflow's own JS code nodes)
# ---------------------------------------------------------------------------
def build_offer_readable_text(offer: Optional[dict]) -> str:
    """Python port of the `parsing offer to text` code node."""
    if not offer:
        return "The user does not select the offer suggested by AI"

    def line(label: str, value) -> str:
        if value and value != "" and value != "0":
            return f"{label} {value}\n"
        return ""

    g = offer.get
    text = ""

    # Header
    text += "ลูกค้าสนใจเข้าร่วมมาตรการแก้ไขหนี้ที่ระบบ Smart ทันหนี้แนะนำผ่าน"
    text += f"แผน{g('plan_desc', '')}\n"
    text += line("NCB Badge", g("ncb_badge"))
    text += line("โดยระบบพิจารณาเสนอแนวทางตาม", g("source_desc"))
    text += "─" * 35 + "\n\n"

    # Accounts Summary
    text += "โดยบัญชีที่ต้องการนำมาพิจารณาเข้าร่วมโครงการได้แก่\n"
    text += f"   บัญชีเลขที่: {g('accounts', '')}\n"
    text += f"   ซึ่งนับเป็น {g('cnt_eligible', '')} บัญชีจาก {g('cnt_total', '')} บัญชีของธนาคาร\n"
    text += line("   ยอดหนี้รวม", f"{g('total_os', '')} บาท")
    text += "\n"

    # Financial Summary
    text += "โดยภาพรวมของการปรับเปลี่ยนการผ่อนชำระ\n"
    text += line("   ขอปรับเปลี่ยนค่างวดรวมเดิม", f"{g('prev_inst', '')} บาท")
    text += line("   เป็นค่างวดใหม่", f"{g('new_inst', '')} บาท")
    text += "   โดยเพิ่มค่างวดในปีที่ 2 และ 3\n" if g("step_label") else ""
    text += "\n"

    # Offer Terms
    text += line("   อัตราดอกเบี้ยที่พิจารณาเพื่อเปิดบัญชีสินเชื่อใหม่ ด้วยอัตรา", g("int_rate_new"))
    text += line("   ระยะเวลาผ่อนชำระตามสัญญาเดิมที่", g("term_actual_old"))
    text += line("   ระยะเวลาผ่อนชำระที่ต้องการให้พิจารณาที่", g("term_remain_new"))
    text += line("   ระยะเวลาผ่อนชำระมีการเปลี่ยนแปลงรวมที่", g("term_change"))
    text += line("   โดยมีค่างวดปีที่ 2-3 ที่", g("inst_y2y3"))
    text += line("   โดยมีค่างวดภายหลังจาก 3 เดือนที่", g("inst_after_3m"))
    text += line("   คาดการณ์การเปลี่ยนแปลงของดอกเบี้ยรวม อยู่ที่", g("int_total_change"))
    text += "\n"

    # Balloon Rows
    balloon_rows = g("balloon_rows") or []
    if len(balloon_rows) > 0:
        text += "โดยมีค่าชำระสินเชื่อเบ็ดเสร็จ ณ วันสิ้นสุดสัญญา\n"
        for row in balloon_rows:
            parts = row.split("|")
            acc_no = parts[0] if len(parts) > 0 else ""
            term = parts[1] if len(parts) > 1 else ""
            payment = parts[2] if len(parts) > 2 else ""
            text += f"   บัญชี {acc_no} | {term} งวด | ชำระ {payment} บาท\n"
    text += "\n"

    # Account Details
    account_details = g("account_details") or []
    if len(account_details) > 0:
        text += "รายละเอียดการเปลี่ยนแปลงของสินเชื่อรายบัญชีที่ลูกค้าต้องการ\n"
        for i, acc in enumerate(account_details):
            ag = acc.get
            text += f"\n   [{i + 1}] {ag('acc_name', '')}\n"
            text += line("       บัญชีเลขที่", ag("acc_no"))
            text += line("       ยอดหนี้คงเหลือ", f"{ag('os', '')} บาท")
            text += line("       อัตราดอกเบี้ย", f"{ag('int_rate', '')}% ต่อปี")
            text += line("       จำนวนงวดเดิม", ag("term_old"))
            text += line("       การเปลี่ยนแปลงจำนวนงวด", ag("term_change"))
            text += line("       ค่างวดเดิม", ag("inst_old"))
            text += line("       ค่างวดใหม่", ag("inst_change"))
            text += line("       ค่างวดปีที่ 1 (ที่เปลี่ยน)", ag("inst_change_y1"))
            text += line("       ค่างวดปีที่ 2-3", ag("inst_y2y3"))
            text += line("       ค่างวดหลัง 3 เดือน", ag("inst_after_3m"))
            text += line("       ค่างวดสินเชื่อใหม่", ag("inst_new_loan"))
            text += line("       ค่าชำระสินเชื่อเบ็ดเสร็จ ณ วันสิ้นสุดสัญญา", ag("balloon_payment"))
            text += line("       ดอกเบี้ยรวมเดิม", ag("int_total_old"))
            text += line("       คาดการณ์การเปลี่ยนแปลงของดอกเบี้ยรวม", ag("int_total_change"))
            text += line("       หมายเหตุ (ไม่มีสิทธิ์)", ag("inelig_note"))
        text += "\n"

    # Notes
    notes = g("notes") or []
    if len(notes) > 0:
        text += "หมายเหตุ\n"
        for i, note in enumerate(notes):
            text += f"   {i + 1}. {note}\n"

    return text


def build_user_message_summary(history: list) -> str:
    """Python port of the `parsing history` code node."""
    user_msgs = [m for m in history if m.get("role") == "USER" and m.get("type") == "text"]
    user_msgs.sort(key=lambda m: m.get("createdAt", ""))
    return "\n".join(f"message {i + 1}: {m.get('content', '')}" for i, m in enumerate(user_msgs))


def build_age_range(age: int) -> str:
    """Python port of the `ageRange` calculation inside `prepare_input_to_summarise`."""
    base = (age // 5) * 5
    return f"{base}-{base + 4} years old"


# ---------------------------------------------------------------------------
# Payload model
# ---------------------------------------------------------------------------
@dataclass
class SummaryWebhookPayload:
    rawUserMessage: str  # latest customer message (used only if no offer was selected)
    selectedOffer: list = field(default_factory=list)  # [] or [offerCard dict]
    lastAIMessageContent: str = ""
    history: list = field(default_factory=list)  # [{role, type, content, createdAt}, ...]
    narrative: str = ""
    preference: str = ""
    maxPaymentRaw: float = 0  # 0 -> "Not provided"
    age: int = 0
    employmentType: str = ""
    monthsAsCustomer: int = 0
    dpd: int = 0
    sumOsNCB: float = 0
    installmentNCB_Y1: float = 0
    installmentNCB_Y2: float = 0
    installmentNCB_Y3: float = 0

    def to_dict(self) -> dict:
        has_offer = bool(self.selectedOffer)
        offer = self.selectedOffer[0] if has_offer else None
        return {
            "userMessage": (
                "The user select one of the offer provided by AI" if has_offer else self.rawUserMessage
            ),
            "LastAImessage": self.lastAIMessageContent,
            "userMessageSummary": build_user_message_summary(self.history),
            "narrative": self.narrative,
            "preference": self.preference,
            "maxPayment": "Not provided" if self.maxPaymentRaw == 0 else f"{self.maxPaymentRaw} THB",
            "ageRange": build_age_range(self.age),
            "employmentType": self.employmentType,
            "monthsAsCustomer": self.monthsAsCustomer,
            "dpd": self.dpd,
            "sumOsNCB": self.sumOsNCB,
            "installmentNCB_Y1": self.installmentNCB_Y1,
            "installmentNCB_Y2": self.installmentNCB_Y2,
            "installmentNCB_Y3": self.installmentNCB_Y3,
            "offerReadableText": build_offer_readable_text(offer),
        }


def call_summary(payload: SummaryWebhookPayload, timeout: int = 60) -> dict:
    """POST a payload to the Summary webhook and return the parsed JSON response."""
    url = get_webhook_url()
    response = requests.post(url, json=payload.to_dict(), timeout=timeout)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Example payloads
# ---------------------------------------------------------------------------

# 1) Baseline — reproduces the workflow's own pinned Webhook example exactly.
#    Customer selected an MOU debt-consolidation offer.
BASELINE_OFFER = {
    "plan_id": "PLMOU0220260608161328",
    "plan_desc": "รวมหนี้ผ่านสินเชื่อ MOU แบบไม่มีหลักประกัน",
    "ncb_badge": "",
    "accounts": "10000005, 10000006",
    "cnt_eligible": "2",
    "cnt_total": "2",
    "total_os": "102,000.00",
    "prev_inst": "5,100",
    "new_inst": "3,200",
    "step_label": "",
    "source_desc": "ความสามารถในการชำระ",
    "int_rate_new": "6.00% ต่อปี (ข้าราชการ)",
    "term_actual_old": "",
    "term_remain_new": "",
    "term_change": "36 → 35 งวด",
    "inst_y2y3": "",
    "inst_after_3m": "",
    "int_total_change": "20,151.32 → 9,388.93 บาท",
    "balloon_rows": [],
    "notes": [
        "มาตรการนี้เป็นคำแนะนำสำหรับการเปิดสินเชื่อใหม่ การพิจารณาจะเป็นไปตามระเบียบการอนุมัติสินเชื่อของธนาคาร",
        "มาตรการนี้สำหรับบุคลากรขององค์กรที่มีการลงนาม MOU กับธนาคารเท่านั้น",
    ],
    "account_details": [
        {
            "acc_no": "10000005,10000006",
            "acc_name": "สินเชื่อส่วนบุคคลอเนกประสงค์ภายใต้ MOU สำหรับรวมสินเชื่อ",
            "os": "102,000.00",
            "int_rate": "6.00",
            "term_old": "",
            "term_change": "35 งวด",
            "inst_old": "",
            "inst_change": "",
            "inst_change_y1": "",
            "inst_y2y3": "",
            "inst_after_3m": "",
            "inst_new_loan": "3,200 บาท",
            "balloon_payment": "",
            "int_total_old": "",
            "int_total_change": "20,151.32 → 9,388.93 บาท",
            "inelig_note": "",
        }
    ],
}

BASELINE_HISTORY = [
    {
        "role": "USER",
        "sessionId": "457f",
        "content": "ต้องการปรับโครงสร้างหนี้สินเชื่อส่วนบุคคล\n",
        "agentUsed": "UserMessage",
        "type": "text",
        "createdAt": "2026-06-08T15:14:47.149Z",
    },
    {
        "role": "BOT",
        "sessionId": "457f",
        "content": "สวัสดีค่ะ ยินดีต้อนรับเข้าสู่ระบบ KTB Care ที่จะช่วยตอบคำถามและให้คำปรึกษาเกี่ยวกับปัญหาหนี้ ลูกค้าสามารถระบุปัญหาที่ต้องการปรึกษามาได้เลยค่ะ",
        "agentUsed": "AutomatedMessage",
        "type": "text",
        "createdAt": "2026-06-08T15:14:16.277Z",
    },
]

EXAMPLES: dict[str, SummaryWebhookPayload] = {
    "baseline_offer_selected": SummaryWebhookPayload(
        rawUserMessage="(unused — offer was selected)",
        selectedOffer=[BASELINE_OFFER],
        lastAIMessageContent="เสนอแนวทางการปรับปรุงสินเชื่อ ด้วยแนวทางและข้อมูลดังนี้\n\n",
        history=BASELINE_HISTORY,
        narrative="The customer selects the debt solution program of interest and the system proceeds to relevant staff.",
        preference="DebtBurden",
        maxPaymentRaw=0,
        age=52,
        employmentType="ข้าราชการ",
        monthsAsCustomer=60,
        dpd=12,
        sumOsNCB=180000,
        installmentNCB_Y1=3000,
        installmentNCB_Y2=5000,
        installmentNCB_Y3=5000,
    ),
    # 2) CASE 2 — customer did NOT select an offer; instead requests staff
    #    contact directly, citing a hardship. offer_suitability should follow
    #    the "did not accept an offer" branch in the agent's prompt.
    "case2_no_offer_staff_request": SummaryWebhookPayload(
        rawUserMessage="ช่วยให้เจ้าหน้าที่ติดต่อกลับด่วนได้ไหมคะ ตอนนี้ป่วยหนักและขาดรายได้ชั่วคราว",
        selectedOffer=[],
        lastAIMessageContent="รับทราบค่ะ ระบบจะส่งเรื่องให้เจ้าหน้าที่ติดต่อกลับค่ะ",
        history=[
            {
                "role": "USER",
                "sessionId": "demo-002",
                "content": "ตอนนี้ป่วยหนัก รายได้หายไปชั่วคราว อยากให้เจ้าหน้าที่ติดต่อกลับ",
                "type": "text",
                "createdAt": "2026-06-18T09:00:00.000Z",
            },
            {
                "role": "USER",
                "sessionId": "demo-002",
                "content": "ช่วยให้เจ้าหน้าที่ติดต่อกลับด่วนได้ไหมคะ",
                "type": "text",
                "createdAt": "2026-06-18T09:05:00.000Z",
            },
        ],
        narrative="The customer reports a sudden illness causing temporary income loss and explicitly requests urgent staff callback rather than selecting an automated offer.",
        preference="FinancialShock",
        maxPaymentRaw=0,
        age=37,
        employmentType="พนักงานประจำ (Salaried)",
        monthsAsCustomer=24,
        dpd=64,
        sumOsNCB=95000,
        installmentNCB_Y1=4500,
        installmentNCB_Y2=4500,
        installmentNCB_Y3=4500,
    ),
    # 3) Customer states a concrete max payment without selecting an offer —
    #    exercises the non-zero maxPayment formatting ("<amount> THB").
    "case3_maxpayment_stated_no_offer": SummaryWebhookPayload(
        rawUserMessage="ผ่อนได้สูงสุดเดือนละ 2500 บาทครับ ขอให้พิจารณาขยายระยะเวลาผ่อนชำระ",
        selectedOffer=[],
        lastAIMessageContent="รับทราบค่ะ จะนำข้อมูลไปพิจารณาเสนอแนวทางที่เหมาะสมค่ะ",
        history=[
            {
                "role": "USER",
                "sessionId": "demo-003",
                "content": "ผ่อนได้สูงสุดเดือนละ 2500 บาทครับ ขอให้พิจารณาขยายระยะเวลาผ่อนชำระ",
                "type": "text",
                "createdAt": "2026-06-19T10:00:00.000Z",
            }
        ],
        narrative="The customer states a maximum affordable payment of 2,500 THB/month and requests a term-extension review without selecting a specific offer.",
        preference="DebtBurden",
        maxPaymentRaw=2500,
        age=45,
        employmentType="ธุรกิจส่วนตัว (Self-employed)",
        monthsAsCustomer=80,
        dpd=30,
        sumOsNCB=210000,
        installmentNCB_Y1=6000,
        installmentNCB_Y2=6000,
        installmentNCB_Y3=6000,
    ),
    # 4) Different offer shape — exercises the balloon-rows branch and the
    #    "increase in Year 2/3" (step_label) branch in offerReadableText.
    "case4_offer_with_balloon_and_stepup": SummaryWebhookPayload(
        rawUserMessage="(unused — offer was selected)",
        selectedOffer=[
            {
                "plan_id": "PLSTEP0120260619100000",
                "plan_desc": "ปรับโครงสร้างหนี้แบบขั้นบันได (Step-up)",
                "ncb_badge": "Gold",
                "accounts": "1xxxxxxx0974,1xxxxxxx0975",
                "cnt_eligible": "2",
                "cnt_total": "2",
                "total_os": "92,655.57",
                "prev_inst": "5,200",
                "new_inst": "2,000",
                "step_label": "yes",
                "source_desc": "ความสามารถในการชำระและ NCB Segment",
                "int_rate_new": "8.50% ต่อปี",
                "term_actual_old": "41 งวด",
                "term_remain_new": "60 งวด",
                "term_change": "41 → 60 งวด",
                "inst_y2y3": "3,500 บาท",
                "inst_after_3m": "2,500 บาท",
                "int_total_change": "15,200.00 → 18,900.00 บาท",
                "balloon_rows": ["1xxxxxxx0974|60|15000", "1xxxxxxx0975|60|9000"],
                "notes": ["แผนนี้มีการปรับเพิ่มค่างวดในปีที่ 2 และ 3 ตามความสามารถในการชำระที่คาดการณ์"],
                "account_details": [],
            }
        ],
        lastAIMessageContent="น้องฟินขอเสนอแผนปรับโครงสร้างหนี้แบบขั้นบันไดค่ะ",
        history=[
            {
                "role": "USER",
                "sessionId": "demo-004",
                "content": "ขอแผนที่ผ่อนน้อยลงในช่วงแรกได้ไหมครับ",
                "type": "text",
                "createdAt": "2026-06-19T11:00:00.000Z",
            }
        ],
        narrative="The customer requested a lower initial installment; the advisor offered a step-up restructuring plan with balloon payments at contract end.",
        preference="TemporaryCashflow",
        maxPaymentRaw=0,
        age=33,
        employmentType="พนักงานประจำ (Salaried)",
        monthsAsCustomer=40,
        dpd=64,
        sumOsNCB=120000,
        installmentNCB_Y1=3500,
        installmentNCB_Y2=3500,
        installmentNCB_Y3=3500,
    ),
}


def run_examples() -> None:
    for label, payload in EXAMPLES.items():
        print(f"\n=== {label} ===")
        print("Request: ", json.dumps(payload.to_dict(), ensure_ascii=False, indent=2)[:800], "...")
        try:
            result = call_summary(payload)
            print("Response:", json.dumps(result, ensure_ascii=False, indent=2))
        except requests.exceptions.RequestException as exc:
            print(f"Request failed: {exc}")


if __name__ == "__main__":
    run_examples()
