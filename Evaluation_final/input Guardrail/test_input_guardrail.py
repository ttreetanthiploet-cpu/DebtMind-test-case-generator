"""
test_input_guardrail.py

Test client + evaluation harness for the "Input Guardrail" n8n workflow,
using the labeled test cases in test_case_InputGuardrail.xlsx.

WHAT THIS WORKFLOW DOES
--------------------------
Given a single `message` string, the workflow runs two stages:

  1. Pre-Checks (Format + Banlist) — a regex-based code node that blocks
     empty/over-length messages and a hardcoded banlist (profanity, fraud/
     debt-evasion terms, self-harm terms, Thai politics/monarchy terms,
     sexual terms). If this stage blocks the message, the workflow skips
     straight to a "fail" result WITHOUT running the LLM guardrails below.
  2. Guardrails (LangChain guardrails node, only runs if step 1 passed) —
     checks jailbreak, nsfw, PII ("personalData"), and a custom "Banning
     topics" check (covers: sexual content, self-harm, investment/stock/
     crypto/trading advice, Thai politics, the monarchy) via an LLM.

ENTRY POINTS
--------------
- `When Executed by Another Workflow`: only reachable as a sub-workflow.
- `Webhook` (path "d1ad9bfc-8ff7-40f7-95d9-a30346675e52"): reachable over
  HTTP — what this script tests. Unlike the Advisor/Summary workflows in
  this project, this workflow's input is trivially simple (just one field,
  `message`), so `aggregate_webhook_input`'s plain `{...body, inputPath:
  'webhook'}` spread is sufficient — no flattening gap here.

REQUEST PAYLOAD
------------------
    { "message": "<the user's message>" }

RESPONSE PAYLOAD
-------------------
    {
      "fail_inputGuardrail": bool,
      "message": "<echoed back>",
      "preCheckViolations": "<comma-separated, or empty string>",
      "matchedBannedWords": "<comma-separated, or empty string>",
      "personalData": bool,
      "jailbreak": bool,
      "nsfw": bool,
      "banningTopics": bool
    }

IMPORTANT NUANCE: when a message is blocked at the Pre-Checks stage (step 1
above), the Guardrails LLM node never runs, so `personalData`/`jailbreak`/
`nsfw`/`banningTopics` will all come back False for that message even if it
conceptually belongs to one of those categories (e.g. a debt-evasion phrase
caught by the banlist, which the test set may label `banningTopics: True`
even though that category technically only reflects the LLM guardrail's
output). `fail_inputGuardrail` is unaffected by this and should still be
correct. Keep this in mind when interpreting the per-category metrics below
— a category mismatch concentrated in pre-check-blocked rows points at this
architectural quirk rather than a bad LLM guardrail.

URL NOTE
----------
- Test URL  (n8n editor "Listen for test event" open): {base}/webhook-test/{path}
- Production URL (workflow Active, which this one is): {base}/webhook/{path}
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Configuration — edit these for your environment
# ---------------------------------------------------------------------------
N8N_BASE_URL = "https://your-n8n-instance.com"  # <-- change me
WEBHOOK_PATH = "d1ad9bfc-8ff7-40f7-95d9-a30346675e52"
USE_TEST_URL = False  # True -> use the "Listen for test event" URL instead
TEST_CASES_XLSX = "test_case_InputGuardrail.xlsx"
LABEL_COLUMNS = ["personalData", "jailbreak", "nsfw", "banningTopics", "fail_inputGuardrail"]
MAX_WORKERS = 5  # concurrent requests; keep modest to avoid rate-limiting the LLM guardrail


def get_webhook_url() -> str:
    prefix = "webhook-test" if USE_TEST_URL else "webhook"
    return f"{N8N_BASE_URL}/{prefix}/{WEBHOOK_PATH}"


# ---------------------------------------------------------------------------
# Webhook caller
# ---------------------------------------------------------------------------
def call_guardrail(message: str, timeout: int = 30, retries: int = 2) -> dict:
    """POST a single message to the Input Guardrail webhook and return the parsed JSON."""
    url = get_webhook_url()
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, json={"message": message}, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def load_test_cases(path: str = TEST_CASES_XLSX) -> pd.DataFrame:
    df = pd.read_excel(path)
    for col in LABEL_COLUMNS:
        df[col] = df[col].astype(bool)
    return df


def run_test_suite(df: pd.DataFrame, max_workers: int = MAX_WORKERS) -> pd.DataFrame:
    """Calls the webhook for every row and returns a results dataframe with
    expected_* and actual_* columns for each label, plus any request errors."""
    results = [None] * len(df)

    def _run_row(i: int, row: pd.Series) -> tuple[int, dict]:
        try:
            resp = call_guardrail(row["User Input"])
            return i, {f"actual_{col}": bool(resp.get(col, False)) for col in LABEL_COLUMNS} | {
                "preCheckViolations": resp.get("preCheckViolations", ""),
                "matchedBannedWords": resp.get("matchedBannedWords", ""),
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001
            return i, {f"actual_{col}": None for col in LABEL_COLUMNS} | {
                "preCheckViolations": None,
                "matchedBannedWords": None,
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_row, i, row) for i, row in df.iterrows()]
        for fut in as_completed(futures):
            i, res = fut.result()
            results[i] = res

    actual_df = pd.DataFrame(results)
    out = pd.concat([df.reset_index(drop=True), actual_df], axis=1)
    for col in LABEL_COLUMNS:
        out = out.rename(columns={col: f"expected_{col}"})
    return out


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------
def evaluate(results_df: pd.DataFrame, label_columns: list = LABEL_COLUMNS) -> pd.DataFrame:
    """Computes accuracy/precision/recall/F1/confusion-matrix counts for each label."""
    valid = results_df[results_df["error"].isna()]
    skipped = len(results_df) - len(valid)
    if skipped:
        print(f"⚠️  {skipped} row(s) had request errors and were excluded from metrics.")

    rows = []
    for col in label_columns:
        y_true = valid[f"expected_{col}"].astype(bool)
        y_pred = valid[f"actual_{col}"].astype(bool)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

        rows.append(
            {
                "label": col,
                "support": len(valid),
                "positives": int(y_true.sum()),
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else float("nan"),
            }
        )

    metrics_df = pd.DataFrame(rows)

    # Exact-match accuracy: all 5 labels correct simultaneously for a row.
    exact_match = (
        valid[[f"expected_{c}" for c in label_columns]].values
        == valid[[f"actual_{c}" for c in label_columns]].values
    ).all(axis=1)
    print(f"\nExact-match accuracy (all {len(label_columns)} labels correct per row): "
          f"{exact_match.mean():.4f} ({exact_match.sum()}/{len(valid)})")

    return metrics_df


def show_mismatches(results_df: pd.DataFrame, label_columns: list = LABEL_COLUMNS) -> pd.DataFrame:
    """Returns rows where at least one label didn't match, for manual inspection."""
    valid = results_df[results_df["error"].isna()].copy()
    mismatch_mask = pd.Series(False, index=valid.index)
    for col in label_columns:
        mismatch_mask |= valid[f"expected_{col}"] != valid[f"actual_{col}"]
    cols = ["User Input"] + [f"expected_{c}" for c in label_columns] + [f"actual_{c}" for c in label_columns]
    return valid.loc[mismatch_mask, cols]


def run_full_evaluation(xlsx_path: str = TEST_CASES_XLSX) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_test_cases(xlsx_path)
    print(f"Loaded {len(df)} test cases from {xlsx_path}")
    results_df = run_test_suite(df)
    metrics_df = evaluate(results_df)
    print("\n=== Per-label metrics ===")
    print(metrics_df.to_string(index=False))
    mismatches = show_mismatches(results_df)
    print(f"\n{len(mismatches)} row(s) with at least one label mismatch.")
    return results_df, metrics_df


if __name__ == "__main__":
    results_df, metrics_df = run_full_evaluation()
    out_path = Path("guardrail_test_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path.resolve()}")
