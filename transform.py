#!/usr/bin/env python3
"""
transform.py

Python reimplementations of the n8n transformation nodes in Prototype_v1.2,
AdvisorWorkFlow_v1.3, and the Summary Workflow.

These functions let you convert a webapp-level orchestration payload into the
exact input format expected by each AI agent's webhook — without going through
the full n8n orchestration.

Node mapping:
  prepare_context()              → Prototype_v1.2 "Prepare Context"
  to_classification_input()      → Prototype_v1.2 "Parse Classification Input"
  to_advisor_webhook_input()     → AdvisorWorkFlow_v1.3 "parsing input for extractor"
  to_summary_start_input()       → Prototype_v1.2 "summarise_plan to relevant staff json"
                                                or "parsing need_staff_contact"
  to_summary_webhook_input()     → Summary Workflow "prepare_input_to_summarise"
                                   + "parsing offer to text" + "parsing history"
  to_output_guardrail_input()    → Output Guardrail webhook body
                                   (detects & converts raw JSON offer arrays in prevAIMessage)
"""

import json
import math
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_acc_no(acc_no: str) -> str:
    """
    Mirrors the maskAccNo function inside Prepare Context.
    e.g. "1234567890" → "1xxxxx7890"
    """
    s = str(acc_no)
    if len(s) <= 5:
        return s
    return s[0] + "x" * (len(s) - 5) + s[-4:]


# ---------------------------------------------------------------------------
# Node: Prepare Context  (Prototype_v1.2)
# ---------------------------------------------------------------------------


def prepare_context(raw_payload: dict) -> dict:
    """
    Mirrors the 'Prepare Context' Code node in Prototype_v1.2.

    Input: raw webapp payload (sessionId, customerId, message, messageType,
           customer, account, offerSoln, offerSoln_Acc, sessionInfo, history)
    Output: enriched context dict with masked accounts and resolved session state.
    """
    customer = raw_payload.get("customer", {})
    account = raw_payload.get("account") or []
    session_id = raw_payload["sessionId"]
    customer_id = raw_payload["customerId"]
    message = raw_payload["message"]
    message_type = raw_payload.get("messageType", "TEXT")
    offer_soln = raw_payload.get("offerSoln") or []
    offer_soln_acc = raw_payload.get("offerSoln_Acc") or []

    # Sanitize numeric NCB fields
    if customer:
        customer = dict(customer)
        if not isinstance(customer.get("installmentNcbY2"), (int, float)):
            customer["installmentNcbY2"] = 0
        if not isinstance(customer.get("installmentNcbY3"), (int, float)):
            customer["installmentNcbY3"] = 0

    # Default maxPayment = sum of installments
    default_max_payment = sum(
        a.get("installment", 0)
        for a in account
        if isinstance(a.get("installment"), (int, float))
    )

    # Resolve sessionInfo (list or dict)
    raw_si = raw_payload.get("sessionInfo") or []
    if isinstance(raw_si, list):
        session_info = raw_si[0] if raw_si else {}
    elif isinstance(raw_si, dict):
        session_info = raw_si
    else:
        session_info = {}

    consider_account = session_info.get("considerAccount", "")
    max_payment = (
        session_info["maxPayment"]
        if session_info.get("maxPayment") is not None
        else default_max_payment
    )
    max_term = session_info.get("maxTerm") if session_info.get("maxTerm") is not None else 360
    preference = session_info.get("preference", "")
    narrative = session_info.get("narrative", "")

    # Determine LastAImessage
    history = raw_payload.get("history") or []
    bot_history = [h for h in history if h.get("role") == "BOT"]

    default_bot = {
        "role": "BOT",
        "content": "",
        "type": "text",
        "agentUsed": "",
        "createdAt": _now(),
    }

    if not bot_history:
        last_ai_message = default_bot
    else:
        last_ai_message = sorted(
            bot_history, key=lambda h: h.get("createdAt", ""), reverse=True
        )[0]

        # If last BOT message was a JSON offer card, convert to readable Thai text
        if last_ai_message.get("type", "").lower() == "json":
            try:
                items = json.loads(last_ai_message["content"])
                plan_ids = [i.get("planId") for i in items if i.get("planId")]
                matched = [o for o in offer_soln if o.get("planId") in plan_ids]
                parts = []
                for idx, o in enumerate(matched):
                    parts.append(
                        f"แผนที่ {idx + 1}: {o.get('planDesc', '')}\n"
                        f"  - รูปแบบ: {o.get('solutionDesc', '')}\n"
                        f"  - บัญชี: {o.get('refAccNo', '')}\n"
                        f"  - ระยะเวลา: {o.get('term', '')} งวด\n"
                        f"  - ค่างวด: {o.get('installment', '')} บาท\n"
                        f"  - ดอกเบี้ยรวม: {o.get('totalExpInt', '')} บาท"
                    )
                plan_text = "\n\n".join(parts) if parts else str(items)
                last_ai_message = {
                    "role": last_ai_message["role"],
                    "content": f"เสนอแผนการปรับโครงสร้างสินเชื่อ ด้วยแผนงานและข้อมูลดังนี้\n\n{plan_text}",
                    "type": "OfferText",
                    "agentUsed": last_ai_message.get("agentUsed", ""),
                    "createdAt": last_ai_message.get("createdAt", ""),
                }
            except Exception:
                pass

    # Mask account numbers
    masked_account = []
    for acc in account:
        acc_copy = dict(acc)
        if acc_copy.get("accNo") is not None:
            acc_copy["accNo"] = mask_acc_no(str(acc_copy["accNo"]))
        masked_account.append(acc_copy)

    return {
        "sessionId": session_id,
        "customerId": customer_id,
        "message": message,
        "messageType": message_type,
        "customer": customer,
        "ConsiderAccount": consider_account,
        "maxPayment": max_payment,
        "maxTerm": max_term,
        "preference": preference,
        "LastAImessage": last_ai_message,
        "narrative": narrative,
        "offerSoln": offer_soln,
        "offerSoln_Acc": offer_soln_acc,
        "account": masked_account,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Node: Parse Classification Input  (Prototype_v1.2)
# ---------------------------------------------------------------------------


def to_classification_input(prepared: dict) -> dict:
    """
    Mirrors 'Parse Classification Input' in Prototype_v1.2.
    This is the payload sent to the Classification Agent webhook.

    Webhook: dbce5b9e-1397-459a-871a-5b27433f1640
    """
    last_ai = prepared["LastAImessage"]
    return {
        "userMessage": prepared["message"],
        "LastAImessage": last_ai.get("content", ""),
        "PrevAIagent": last_ai.get("agentUsed", ""),
        "narrative": prepared["narrative"],
    }


# ---------------------------------------------------------------------------
# Node: Parse Advisor Input  (Prototype_v1.2)  →  Start trigger (Advisor WF)
# ---------------------------------------------------------------------------


def to_advisor_start_input(prepared: dict, narrative_from_classifier: str = "") -> dict:
    """
    Mirrors 'Parse Advisor Input' in Prototype_v1.2.
    Produces the payload for the Advisor subworkflow Start trigger.
    Pass narrative_from_classifier if you have the classification output.
    """
    return {
        "sessionId": prepared["sessionId"],
        "customerId": prepared["customerId"],
        "userMessage": prepared["message"],
        "LastAImessage": prepared["LastAImessage"],
        "conversationDesc": {
            "consultAcc": prepared["ConsiderAccount"],
            "maxPayment": prepared["maxPayment"],
            "preference": prepared["preference"],
            "maxTerm": prepared["maxTerm"],
            "offerSoln": prepared["offerSoln"],
            "narrative": narrative_from_classifier or prepared["narrative"],
        },
        "userInfo": prepared["customer"],
        "accInfo": prepared["account"],
        "timestamp": _now(),
    }


# ---------------------------------------------------------------------------
# Node: parsing input for extractor  (AdvisorWorkFlow_v1.3)
# ---------------------------------------------------------------------------


def to_advisor_webhook_input(prepared: dict, narrative_from_classifier: str = "") -> dict:
    """
    Mirrors the 'parsing input for extractor' Code node in AdvisorWorkFlow_v1.3.

    This is the actual format the Advisor webhook expects when called directly
    (not via subworkflow). It pre-processes the raw payload so the
    Debt Solution Extractor LLM receives structured text/JSON strings.

    Webhook: b7607735-bac3-4965-8c68-2ed0ef38aec4
    """
    narrative = narrative_from_classifier or prepared["narrative"]
    offer_soln = prepared.get("offerSoln") or []

    # Build conversationDesc (mirrors enrichedConversationDesc in JS)
    conversation_desc: dict = {
        "consultAcc": prepared["ConsiderAccount"],
        "maxPayment": prepared["maxPayment"],
        "preference": prepared["preference"],
        "maxTerm": prepared["maxTerm"],
        "offerSoln": offer_soln,
        "narrative": narrative,
    }

    # Parse preference JSON if present, merge into conversation_desc
    pref_str = prepared.get("preference", "")
    if pref_str:
        try:
            pref_obj = json.loads(pref_str)
            conversation_desc = {**conversation_desc, **pref_obj}
        except Exception:
            pass

    consult_acc = conversation_desc.get("consultAcc") or ""
    debt_situation = conversation_desc.get("DebtSituation")
    max_payment = conversation_desc.get("maxPayment")
    max_term = conversation_desc.get("maxTerm")
    ref_plan_id = conversation_desc.get("refPlanID")
    max_payment_y2 = conversation_desc.get("maxPaymentY2")
    max_payment_y3 = conversation_desc.get("maxPaymentY3")

    # Build compact account list for the extractor prompt
    acc_used = [
        {
            "account_number": acc.get("accNo"),
            "port": acc.get("port"),
            "creditlimit": acc.get("creditLimit"),
            "os": acc.get("os"),
            "installment": acc.get("installment"),
            "remaining_term": acc.get("remainTerm"),
        }
        for acc in prepared.get("account") or []
    ]

    # LastAImessage is JSON-stringified content string (mirrors JS JSON.stringify)
    last_ai_content = prepared["LastAImessage"].get("content", "")
    last_ai_str = json.dumps(last_ai_content, ensure_ascii=False)

    return {
        "consultAcc": consult_acc,
        "DebtSituation": debt_situation,
        "maxPayment": max_payment,
        "maxTerm": max_term,
        "refPlanID": ref_plan_id,
        "maxPaymentY2": max_payment_y2,
        "maxPaymentY3": max_payment_y3,
        "narrative": narrative,
        "accText": json.dumps(acc_used, ensure_ascii=False),
        "offerText": json.dumps(offer_soln, ensure_ascii=False),
        "accInfotoExtr": acc_used,
        "LastAImessage": last_ai_str,
        "userMessage": prepared["message"],
    }


# ---------------------------------------------------------------------------
# Node: summarise_plan / parsing need_staff_contact  (Prototype_v1.2)
# ---------------------------------------------------------------------------


def to_summary_start_input(
    prepared: dict,
    offer_card: Optional[dict],
    narrative: str = "",
) -> dict:
    """
    Builds the Summary Workflow Start-trigger input.

    If offer_card is provided: mirrors 'summarise_plan to relevant staff json'
      (user selected an offer card — messageType=json path).
    If offer_card is None: mirrors 'parsing need_staff_contact'
      (user sent text message routed to summary — staff contact path).
    """
    if offer_card is not None:
        selected_offer = [offer_card]
        conv_narrative = (
            "The customer selects the debt solution program of interest "
            "and the system proceeds to relevant staff."
        )
    else:
        selected_offer = []
        conv_narrative = (
            narrative
            or "The customer request to pass their request to relevant staff to contact back."
        )

    return {
        "sessionId": prepared["sessionId"],
        "customerId": prepared["customerId"],
        "userMessage": prepared["message"],
        "messageType": prepared["messageType"],
        "LastAImessage": prepared["LastAImessage"],
        "selectedOffer": selected_offer,
        "conversationDesc": {
            "consultAcc": prepared["ConsiderAccount"],
            "maxPayment": prepared["maxPayment"],
            "preference": prepared["preference"],
            "maxTerm": prepared["maxTerm"],
            "offerSoln": prepared["offerSoln"],
            "narrative": conv_narrative,
        },
        "userInfo": prepared["customer"],
        "accInfo": prepared["account"],
        "history": prepared["history"],
        "timestamp": _now(),
    }


# ---------------------------------------------------------------------------
# Helper: format offer card to Thai readable text
# Mirrors 'parsing offer to text' inside Summary Workflow
# ---------------------------------------------------------------------------


def format_offer_to_text(offer: Optional[dict]) -> str:
    """
    Converts a selectedOffer[0] dict (from the Python advisor API response)
    into Thai-readable text for the Summary Agent prompt.

    Mirrors the 'parsing offer to text' Code node in Summary Workflow.
    """
    if not offer:
        return "ไม่มีข้อมูลแผนที่ลูกค้าเลือก"

    lines: list[str] = []

    # Header
    badge = offer.get("ncb_badge", "")
    badge_txt = f" [{badge}]" if badge else ""
    lines.append(f"แผน: {offer.get('plan_desc', '')}{badge_txt}")
    lines.append(f"ที่มาของข้อเสนอ: {offer.get('source_desc', '')}")
    lines.append("")

    # Accounts summary
    lines.append(f"บัญชีที่เข้าร่วม: {offer.get('accounts', '')}")
    lines.append(
        f"จำนวนบัญชีที่เข้าเงื่อนไข: {offer.get('cnt_eligible', '')} / "
        f"{offer.get('cnt_total', '')} บัญชี"
    )
    lines.append(f"ยอดหนี้รวม: {offer.get('total_os', '')} บาท")
    lines.append("")

    # Payment
    lines.append(f"ค่างวดเดิม: {offer.get('prev_inst', '')} บาท/เดือน")
    lines.append(f"ค่างวดใหม่: {offer.get('new_inst', '')} บาท/เดือน")

    step_label = offer.get("step_label", "")
    if step_label:
        lines.append(f"ข้อมูล Step-up: {step_label}")

    inst_y2y3 = offer.get("inst_y2y3", "")
    if inst_y2y3:
        lines.append(f"ค่างวดปีที่ 2-3: {inst_y2y3}")

    inst_after_3m = offer.get("inst_after_3m", "")
    if inst_after_3m:
        lines.append(f"ค่างวดหลัง 3 เดือน: {inst_after_3m}")

    lines.append("")

    # Term & interest
    lines.append(f"อัตราดอกเบี้ย: {offer.get('int_rate_new', '')}")
    lines.append(f"จำนวนงวด: {offer.get('term_change', '')}")

    int_total = offer.get("int_total_change", "")
    if int_total:
        lines.append(f"ดอกเบี้ยรวมตลอดสัญญา: {int_total}")

    # Balloon rows
    balloon_rows = offer.get("balloon_rows") or []
    if balloon_rows:
        lines.append("")
        lines.append("รายละเอียด Balloon:")
        for row in balloon_rows:
            lines.append(f"  - {row}")

    # Account details
    acc_details = offer.get("account_details") or []
    if acc_details:
        lines.append("")
        lines.append("รายละเอียดรายบัญชี:")
        for acc in acc_details:
            acc_no = acc.get("accNo") or acc.get("acc_no") or ""
            inst = acc.get("installment", "")
            term = acc.get("term", "")
            lines.append(f"  บัญชี {acc_no}: ค่างวด {inst} บาท, {term} งวด")

    # Notes
    notes = offer.get("notes") or []
    if notes:
        lines.append("")
        lines.append("หมายเหตุ:")
        for note in notes:
            note_text = note if isinstance(note, str) else note.get("text", str(note))
            lines.append(f"  * {note_text}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: aggregate user messages from history
# Mirrors 'parsing history' inside Summary Workflow
# ---------------------------------------------------------------------------


def aggregate_user_messages(history: list) -> str:
    """
    Joins all USER messages from history as "message N: <content>".
    Mirrors 'parsing history' Code node in Summary Workflow.
    """
    user_msgs = [h for h in history if h.get("role") == "USER"]
    if not user_msgs:
        return ""
    return "\n".join(
        f"message {i + 1}: {h.get('content', '')}"
        for i, h in enumerate(user_msgs)
    )


# ---------------------------------------------------------------------------
# Node: prepare_input_to_summarise  (Summary Workflow)
# ---------------------------------------------------------------------------


def _remap_user_info_for_summary(user_info: dict) -> dict:
    """
    Applies the same field renames that 'parsing to Offer Engine Python'
    in AdvisorWorkFlow_v1.3 does to userInfo before Summary Workflow receives it.
    Accepts either raw (camelCase) or already-remapped (PascalCase) format.
    """
    # Detect format by checking which keys exist
    if "CustomerSegment" in user_info:
        return user_info  # already remapped
    return {
        "cif": str(user_info.get("cif", "")),
        "mob": user_info.get("mob"),
        "CustomerSegment": user_info.get("customerSegment"),
        "DebtMindSegment": user_info.get("debtMindSegment"),
        "grpDPD": user_info.get("grpDpd"),
        "SumOsNCB": user_info.get("sumOsNcb"),
        "NCBCheckDate": "NA",
        "EligibleProgram": user_info.get("eligibleProgram"),
        "IncomeFromSystem": user_info.get("incomeFromSystem"),
        "name": "NA",
        "age": user_info.get("age"),
        "employment_type": user_info.get("employmentType"),
        "InstallmentNCB_Y1": user_info.get("installmentNcbY1"),
        "InstallmentNCB_Y2": (
            user_info.get("installmentNcbY2")
            if isinstance(user_info.get("installmentNcbY2"), (int, float))
            else 0
        ),
        "InstallmentNCB_Y3": (
            user_info.get("installmentNcbY3")
            if isinstance(user_info.get("installmentNcbY3"), (int, float))
            else 0
        ),
    }


def to_summary_webhook_input(summary_start: dict) -> dict:
    """
    Produces the payload for the Summary Agent webhook when called directly.

    Mirrors the full chain inside Summary Workflow:
      'parsing history' + 'parsing offer to text' + 'prepare_input_to_summarise'

    Webhook: 515736a7-e9eb-4ea0-81cb-bfd4a4248a8b

    summary_start: output of to_summary_start_input()
    """
    conversation_desc = summary_start.get("conversationDesc", {})
    user_info_raw = summary_start.get("userInfo") or {}
    history = summary_start.get("history") or []
    selected_offer = summary_start.get("selectedOffer") or []
    last_ai = summary_start.get("LastAImessage") or {}

    # Remap userInfo to PascalCase (what Summary WF expects)
    ui = _remap_user_info_for_summary(user_info_raw)

    # Age range: floor(age/5)*5 – floor(age/5)*5+4
    age = ui.get("age") or 0
    age_bucket = math.floor(age / 5) * 5
    age_range = f"{age_bucket}-{age_bucket + 4} years old"

    # maxPayment formatting
    mp = conversation_desc.get("maxPayment", 0)
    max_payment_str = "Not provided" if not mp else f"{mp} THB"

    # Aggregate user messages
    user_message_summary = aggregate_user_messages(history)

    # Format offer text
    offer = selected_offer[0] if selected_offer else None
    offer_readable_text = format_offer_to_text(offer)

    # LastAImessage content string
    last_ai_content = (
        last_ai.get("content", "") if isinstance(last_ai, dict) else str(last_ai)
    )

    return {
        "inputPath": "webhook",
        "userMessage": summary_start.get("userMessage", ""),
        "LastAImessage": last_ai_content,
        "userMessageSummary": user_message_summary,
        "narrative": conversation_desc.get("narrative", ""),
        "preference": conversation_desc.get("preference", ""),
        "maxPayment": max_payment_str,
        "ageRange": age_range,
        "employmentType": ui.get("employment_type", ""),
        "monthsAsCustomer": ui.get("mob"),
        "dpd": ui.get("grpDPD"),
        "sumOsNCB": ui.get("SumOsNCB"),
        "installmentNCB_Y1": ui.get("InstallmentNCB_Y1"),
        "installmentNCB_Y2": ui.get("InstallmentNCB_Y2"),
        "installmentNCB_Y3": ui.get("InstallmentNCB_Y3"),
        "offerReadableText": offer_readable_text,
    }


# ---------------------------------------------------------------------------
# Output Guardrail webhook input
# ---------------------------------------------------------------------------


def to_output_guardrail_input(
    reply_message: str,
    user_message: str,
    prev_ai_message: str,
    offer_soln: Optional[list] = None,
) -> dict:
    """
    Builds the payload for the Output Guardrail webhook.

    Webhook: efab08a3-b42d-45e4-b424-99ea86faa364

    Expected output fields:
      fail_outputGuardrail  : bool
      preCheckViolations    : str  (comma-joined, e.g. "empty_output")
      personalData          : bool
      nsfw                  : bool
      hallucinationHarm     : bool

    prevAIMessage handling:
      In the normal n8n flow, Prepare Context already converts JSON offer arrays
      to Thai text before the guardrail sees them. When testing via webhook
      directly you may pass either:
        - Plain Thai text (already converted) → passed as-is
        - Raw JSON offer array string         → auto-converted to Thai text here

    The conversion logic mirrors Prepare Context's JSON-offer branch.
    """
    formatted_prev = prev_ai_message or ""

    # Detect raw JSON offer array (type="json" bot message content)
    stripped = formatted_prev.strip()
    if stripped.startswith("["):
        try:
            items = json.loads(stripped)
            if isinstance(items, list) and items and isinstance(items[0], dict):
                offer_texts: list[str] = []
                for item in items:
                    if item.get("jsonType") == "offerCard":
                        offer_data = item.get("offerCard") or item
                        # If a full offer_soln list was supplied, look up complete details
                        plan_id = item.get("planId")
                        if plan_id and offer_soln:
                            full = next(
                                (o for o in offer_soln if o.get("planId") == plan_id),
                                None,
                            )
                            if full:
                                offer_data = full
                        offer_texts.append(format_offer_to_text(offer_data))
                if offer_texts:
                    formatted_prev = (
                        "เสนอแผนการปรับโครงสร้างสินเชื่อ ด้วยแผนงานและข้อมูลดังนี้\n\n"
                        + "\n\n---\n\n".join(offer_texts)
                    )
        except (json.JSONDecodeError, Exception):
            pass  # not valid JSON; pass raw string through

    return {
        "replyMessage": reply_message,
        "prevAIMessage": formatted_prev,
        "UserMessage": user_message,
    }
