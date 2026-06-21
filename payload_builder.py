"""
payload_builder.py

Builds random orchestration payloads — the exact JSON body the webapp
POSTs to the Prototype_v1.2 n8n webhook — using real database samples.

Scenario types generated:
  NEW_CONVERSATION      Fresh session, no history, no offers
  AFTER_OFFER_TEXT      Offers already shown; customer replies via text
  OFFER_SELECTED        Customer selects one offer (messageType=JSON)
  NEGOTIATE_PAYMENT     Customer wants to lower installment after seeing offers
  NEGOTIATE_TERM        Customer wants shorter/longer term after seeing offers
  STAFF_REQUEST         Customer asks to be contacted by staff
  EDUCATION_QUESTION    Customer asks an educational question
  OFF_TOPIC             Customer message unrelated to debt restructuring
  MULTI_TURN            Mid-conversation with several history turns
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from typing import Optional

from .db_sampler import DatabaseSampler

# ── Scenario registry ─────────────────────────────────────────────────────────

SCENARIOS = [
    "NEW_CONVERSATION",
    "AFTER_OFFER_TEXT",
    "OFFER_SELECTED",
    "NEGOTIATE_PAYMENT",
    "NEGOTIATE_TERM",
    "STAFF_REQUEST",
    "EDUCATION_QUESTION",
    "OFF_TOPIC",
    "MULTI_TURN",
]

# Weighted distribution — some scenarios more common than others
SCENARIO_WEIGHTS = [12, 15, 10, 12, 8, 10, 8, 5, 20]

# ── Static example messages used as placeholders before Gemini fills them ─────
# Gemini will replace these with realistic messages derived from context.
_PLACEHOLDER = "__GEMINI_WILL_FILL__"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _session_id() -> str:
    return f"test-session-{random.randint(100000, 999999)}"


def _history_turn(
    user_msg: str,
    bot_msg: str,
    bot_type: str = "text",
    agent_used: str = "ADVISOR",
    base_ts: str = "2026-06-21T09:00:00Z",
    offset_min: int = 0,
) -> list[dict]:
    from datetime import timedelta
    dt = datetime.fromisoformat(base_ts.replace("Z", "+00:00"))
    user_ts = (dt + timedelta(minutes=offset_min)).isoformat()
    bot_ts = (dt + timedelta(minutes=offset_min + 1)).isoformat()
    return [
        {"role": "USER", "content": user_msg, "agentUsed": "UserMessage",
         "type": "text", "createdAt": user_ts},
        {"role": "BOT", "content": bot_msg, "agentUsed": agent_used,
         "type": bot_type, "createdAt": bot_ts},
    ]


class PayloadBuilder:
    def __init__(self, sampler: DatabaseSampler):
        self.sampler = sampler

    def build(self, scenario: Optional[str] = None) -> dict:
        """Return a complete orchestration payload for the given scenario.
        If scenario is None, one is chosen randomly by weight."""
        if scenario is None:
            scenario = random.choices(SCENARIOS, weights=SCENARIO_WEIGHTS, k=1)[0]

        builder = getattr(self, f"_build_{scenario.lower()}")
        payload = builder()
        payload["scenario"] = scenario
        return payload

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pick_customer_and_accounts(self, segment: Optional[str] = None):
        cust = self.sampler.random_customer(segment)
        accounts = self.sampler.get_accounts(cust["cif"])
        # Limit to a reasonable subset (1-3 accounts)
        if len(accounts) > 3:
            accounts = random.sample(accounts, random.randint(1, 3))
        return cust, accounts

    def _pick_session_with_offers(self):
        """Return (session, offer_summaries, offer_acc_records) for a session that has offers."""
        sessions_with_offers = [
            s for s in self.sampler.sessions
            if self.sampler.offers_by_session.get(s["sessionId"])
        ]
        if not sessions_with_offers:
            return None, [], []
        sess = random.choice(sessions_with_offers)
        sid = sess["sessionId"]
        summaries = self.sampler.get_offers_for_session(sid)
        acc_records = self.sampler.get_offer_accounts_for_session(sid)
        return sess, summaries, acc_records

    def _build_offer_soln(self, summaries: list[dict], acc_records: list[dict]) -> list[dict]:
        return [
            self.sampler.build_offer_soln_entry(s, acc_records)
            for s in summaries
        ]

    def _session_info_base(self, sess: dict) -> dict:
        return {
            "maxTerm": sess.get("maxTerm", 360),
            "maxPayment": sess.get("maxPayment", 0),
            "preference": sess.get("preference", ""),
            "narrative": sess.get("narrative", ""),
            "considerAccount": sess.get("considerAccount", ""),
        }

    def _base_payload(
        self,
        cust: dict,
        accounts: list[dict],
        message: str,
        message_type: str = "TEXT",
        history: Optional[list] = None,
        session_info: Optional[dict] = None,
        offer_soln: Optional[list] = None,
    ) -> dict:
        return {
            "sessionId": _session_id(),
            "customerId": cust["cif"],
            "message": message,
            "messageType": message_type,
            "customer": cust,
            "account": accounts,
            "offerSoln": offer_soln or [],
            "offerSoln_Acc": [],
            "sessionInfo": session_info or {},
            "history": history or [],
        }

    # ── Scenario builders ─────────────────────────────────────────────────────

    def _build_new_conversation(self) -> dict:
        segment = random.choice(self.sampler.segments)
        cust, accounts = self._pick_customer_and_accounts(segment)
        session_info = {
            "maxTerm": 360,
            "maxPayment": 0,
            "preference": "",
            "narrative": "",
            "considerAccount": "",
        }
        return self._base_payload(
            cust, accounts,
            message=_PLACEHOLDER,
            session_info=session_info,
        )

    def _build_after_offer_text(self) -> dict:
        sess, summaries, acc_records = self._pick_session_with_offers()
        offer_soln = self._build_offer_soln(summaries, acc_records)

        segment = random.choice(self.sampler.segments)
        cust, accounts = self._pick_customer_and_accounts(segment)

        # Build offer card JSON text (what the BOT sent)
        offer_json_content = json.dumps([
            {"jsonType": "offerCard", "planId": o["planId"], "offerCard": o}
            for o in offer_soln
        ], ensure_ascii=False)

        history = _history_turn(
            user_msg="ขอดูแผนปรับโครงสร้างหนี้ครับ",
            bot_msg=offer_json_content,
            bot_type="json",
            agent_used="ADVISOR",
        )

        session_info = self._session_info_base(sess) if sess else {
            "maxTerm": 360, "maxPayment": 0, "preference": "", "narrative": "", "considerAccount": ""
        }

        return self._base_payload(
            cust, accounts,
            message=_PLACEHOLDER,
            session_info=session_info,
            history=history,
            offer_soln=offer_soln,
        )

    def _build_offer_selected(self) -> dict:
        sess, summaries, acc_records = self._pick_session_with_offers()
        offer_soln = self._build_offer_soln(summaries, acc_records)

        segment = random.choice(self.sampler.segments)
        cust, accounts = self._pick_customer_and_accounts(segment)

        # Pick one offer to select
        selected = random.choice(offer_soln)
        message = self.sampler.build_offer_card_message(selected)

        offer_json_content = json.dumps([
            {"jsonType": "offerCard", "planId": o["planId"], "offerCard": o}
            for o in offer_soln
        ], ensure_ascii=False)

        history = _history_turn(
            user_msg="สนใจแผนที่เสนอครับ",
            bot_msg=offer_json_content,
            bot_type="json",
            agent_used="ADVISOR",
        )

        session_info = self._session_info_base(sess) if sess else {
            "maxTerm": 360, "maxPayment": 0, "preference": "", "narrative": "", "considerAccount": ""
        }

        return self._base_payload(
            cust, accounts,
            message=message,
            message_type="JSON",
            session_info=session_info,
            history=history,
            offer_soln=offer_soln,
        )

    def _build_negotiate_payment(self) -> dict:
        sess, summaries, acc_records = self._pick_session_with_offers()
        offer_soln = self._build_offer_soln(summaries, acc_records)

        segment = random.choice(["C1-X", "C2", "C3"])
        cust, accounts = self._pick_customer_and_accounts(segment)

        ref_offer = random.choice(offer_soln) if offer_soln else None
        current_inst = ref_offer["installment"] if ref_offer else 15000
        desired_inst = round(current_inst * random.uniform(0.4, 0.75), -2)

        offer_json_content = json.dumps([
            {"jsonType": "offerCard", "planId": o["planId"], "offerCard": o}
            for o in offer_soln
        ], ensure_ascii=False)

        history = _history_turn(
            user_msg="สอบถามเรื่องแผนปรับโครงสร้างครับ",
            bot_msg=offer_json_content,
            bot_type="json",
            agent_used="ADVISOR",
        )

        pref_obj = {"DebtSituation": "DebtBurden"}
        if ref_offer:
            pref_obj["refPlanID"] = ref_offer["planId"]

        session_info = {
            "maxTerm": sess.get("maxTerm", 360) if sess else 360,
            "maxPayment": int(desired_inst),
            "preference": json.dumps(pref_obj, ensure_ascii=False),
            "narrative": sess.get("narrative", "") if sess else "",
            "considerAccount": sess.get("considerAccount", "") if sess else "",
        }

        return self._base_payload(
            cust, accounts,
            message=_PLACEHOLDER,
            session_info=session_info,
            history=history,
            offer_soln=offer_soln,
        )

    def _build_negotiate_term(self) -> dict:
        sess, summaries, acc_records = self._pick_session_with_offers()
        offer_soln = self._build_offer_soln(summaries, acc_records)

        segment = random.choice(["C2", "C3", "Current"])
        cust, accounts = self._pick_customer_and_accounts(segment)

        ref_offer = random.choice(offer_soln) if offer_soln else None
        current_term = ref_offer["term"] if ref_offer else 120
        desired_term = random.choice([
            max(12, current_term - random.randint(12, 60)),
            min(360, current_term + random.randint(12, 60)),
        ])

        offer_json_content = json.dumps([
            {"jsonType": "offerCard", "planId": o["planId"], "offerCard": o}
            for o in offer_soln
        ], ensure_ascii=False)

        history = _history_turn(
            user_msg="ขอดูแผนครับ",
            bot_msg=offer_json_content,
            bot_type="json",
            agent_used="ADVISOR",
        )

        pref_obj = {"DebtSituation": "DebtBurden"}
        if ref_offer:
            pref_obj["refPlanID"] = ref_offer["planId"]

        session_info = {
            "maxTerm": desired_term,
            "maxPayment": sess.get("maxPayment", 0) if sess else 0,
            "preference": json.dumps(pref_obj, ensure_ascii=False),
            "narrative": sess.get("narrative", "") if sess else "",
            "considerAccount": sess.get("considerAccount", "") if sess else "",
        }

        return self._base_payload(
            cust, accounts,
            message=_PLACEHOLDER,
            session_info=session_info,
            history=history,
            offer_soln=offer_soln,
        )

    def _build_staff_request(self) -> dict:
        segment = random.choice(self.sampler.segments)
        cust, accounts = self._pick_customer_and_accounts(segment)

        # May or may not have seen offers
        has_offers = random.random() < 0.5
        offer_soln = []
        history = []

        if has_offers:
            sess, summaries, acc_records = self._pick_session_with_offers()
            offer_soln = self._build_offer_soln(summaries, acc_records)
            offer_json_content = json.dumps([
                {"jsonType": "offerCard", "planId": o["planId"], "offerCard": o}
                for o in offer_soln
            ], ensure_ascii=False)
            history = _history_turn(
                user_msg="สอบถามเรื่องหนี้ครับ",
                bot_msg=offer_json_content,
                bot_type="json",
                agent_used="ADVISOR",
            )
            narrative = sess.get("narrative", "") if sess else ""
        else:
            narrative = ""

        session_info = {
            "maxTerm": 360,
            "maxPayment": 0,
            "preference": "",
            "narrative": narrative,
            "considerAccount": "",
        }

        return self._base_payload(
            cust, accounts,
            message=_PLACEHOLDER,
            session_info=session_info,
            history=history,
            offer_soln=offer_soln,
        )

    def _build_education_question(self) -> dict:
        segment = random.choice(self.sampler.segments)
        cust, accounts = self._pick_customer_and_accounts(segment)

        session_info = {
            "maxTerm": 360,
            "maxPayment": 0,
            "preference": "",
            "narrative": "",
            "considerAccount": "",
        }

        return self._base_payload(
            cust, accounts,
            message=_PLACEHOLDER,
            session_info=session_info,
        )

    def _build_off_topic(self) -> dict:
        segment = random.choice(self.sampler.segments)
        cust, accounts = self._pick_customer_and_accounts(segment)

        session_info = {
            "maxTerm": 360,
            "maxPayment": 0,
            "preference": "",
            "narrative": "",
            "considerAccount": "",
        }

        # Include some history to simulate mid-conversation off-topic
        history = _history_turn(
            user_msg="สวัสดีครับ",
            bot_msg="สวัสดีครับ ยินดีให้บริการ คุณมีคำถามเกี่ยวกับการปรับโครงสร้างหนี้ไหมครับ",
            bot_type="text",
            agent_used="",
        )

        return self._base_payload(
            cust, accounts,
            message=_PLACEHOLDER,
            session_info=session_info,
            history=history,
        )

    def _build_multi_turn(self) -> dict:
        """Use a real session's conversation history to build a mid-conversation payload."""
        sessions_with_conv = [
            sid for sid, turns in self.sampler.conv_by_session.items()
            if len(turns) >= 4
        ]

        if sessions_with_conv:
            sid = random.choice(sessions_with_conv)
            all_turns = self.sampler.get_conversation(sid)
            sess = self.sampler.session_by_id.get(sid, {})
            summaries = self.sampler.get_offers_for_session(sid)
            acc_records = self.sampler.get_offer_accounts_for_session(sid)
            offer_soln = self._build_offer_soln(summaries, acc_records)

            # Slice a random mid-point in the conversation
            # Take first N turns as history (at least 2 turns, leave last turn as current)
            bot_turns = [t for t in all_turns if t["role"] == "BOT"]
            if len(bot_turns) >= 2:
                cut = random.randint(1, len(bot_turns) - 1) * 2
            else:
                cut = 2
            history = all_turns[:cut]

            # Use next USER turn's content if available, else placeholder
            message_type = "TEXT"
            if cut < len(all_turns) and all_turns[cut]["role"] == "USER":
                raw_msg = all_turns[cut]["content"]
                # If the real message is a JSON offerCard selection, keep it as-is with JSON type
                if raw_msg.strip().startswith("{") and "offerCard" in raw_msg:
                    message = raw_msg
                    message_type = "JSON"
                else:
                    # Use the real message only if it is plain text; otherwise placeholder
                    message = raw_msg if not raw_msg.strip().startswith("[") else _PLACEHOLDER
            else:
                message = _PLACEHOLDER

            session_info = {
                "maxTerm": sess.get("maxTerm", 360),
                "maxPayment": sess.get("maxPayment", 0),
                "preference": sess.get("preference", ""),
                "narrative": sess.get("narrative", ""),
                "considerAccount": sess.get("considerAccount", ""),
            }

            # Pick a random customer (session records don't store CIF directly)
            segment = random.choice(self.sampler.segments)
            cust, accounts = self._pick_customer_and_accounts(segment)

            return self._base_payload(
                cust, accounts,
                message=message,
                message_type=message_type,
                session_info=session_info,
                history=history,
                offer_soln=offer_soln,
            )
        else:
            # Fallback: build a synthetic multi-turn
            return self._build_after_offer_text()
