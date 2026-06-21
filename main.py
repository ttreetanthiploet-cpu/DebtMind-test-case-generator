"""
main.py — Generate random n8n orchestration payload test cases.

Each test case is the exact JSON body the webapp POSTs to the
Prototype_v1.2 n8n webhook, enriched with:
  - "testDescription"   brief description of what the test covers
  - "scenario"          the scenario type used to generate it
  - "messageMode"       strict | creative | adversarial
  - "guardrailCategory" (adversarial only) which guardrail attack was used

Message modes
─────────────
  strict      — Gemini writes a message that strictly matches the planned scenario
  creative    — Gemini freely deviates; unexpected but still plausible
  adversarial — Gemini writes a message designed to stress-test the output guardrail
                (jailbreak, data_fishing, nsfw_abusive, hallucination_bait,
                 prompt_injection, social_engineering)

Usage
─────
    # Generate 10 test cases (random mode mix, uses GEMINI_API_KEY env var)
    python3 -m final_test_case_gen.main 10

    # Force a specific message mode for all cases
    python3 -m final_test_case_gen.main 10 --message-mode adversarial

    # Dry-run — no Gemini, static fallback messages
    python3 -m final_test_case_gen.main 10 --no-ai

    # Only one scenario type
    python3 -m final_test_case_gen.main 5 --scenario NEGOTIATE_PAYMENT

    # Custom output directory
    python3 -m final_test_case_gen.main 20 --output-dir ./my_test_cases

Output
──────
    final_test_case_gen/output/TC-0001/payload.json
    final_test_case_gen/output/TC-0002/payload.json
    ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from final_test_case_gen.db_sampler import DatabaseSampler
from final_test_case_gen.payload_builder import PayloadBuilder, SCENARIOS
from final_test_case_gen.gemini_annotator import GeminiAnnotator, MESSAGE_MODES

_DEFAULT_OUTPUT = Path(__file__).parent / "output"
_DB_DIR = Path(__file__).parent / "database"


def generate(
    n: int,
    api_key: str = "",
    output_dir: Path = _DEFAULT_OUTPUT,
    use_ai: bool = True,
    scenario_filter: str | None = None,
    message_mode: str | None = None,
    verbose: bool = True,
) -> list[Path]:
    """
    Generate `n` test-case payload files.

    Parameters
    ----------
    n               : number of test cases to generate
    api_key         : Gemini API key (falls back to GEMINI_API_KEY env var)
    output_dir      : root folder; each case saved as <output_dir>/TC-NNNN/payload.json
    use_ai          : if False, skip Gemini and use static fallback messages
    scenario_filter : if set, only generate this scenario type
    message_mode    : "strict" | "creative" | "adversarial" | None (random by weight)
    verbose         : print progress to stdout

    Returns
    -------
    List of Paths to the generated payload.json files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler = DatabaseSampler(_DB_DIR)
    builder = PayloadBuilder(sampler)
    annotator = GeminiAnnotator(api_key if use_ai else "")

    if verbose:
        ai_status = "Gemini enabled" if annotator.available else "no-AI mode (fallback messages)"
        mode_status = message_mode or "random (strict 50% / creative 30% / adversarial 20%)"
        print(f"Generating {n} test case(s) — {ai_status}")
        print(f"Message mode: {mode_status}")
        print(f"Output: {output_dir.resolve()}\n")

    saved_paths: list[Path] = []

    for i in range(1, n + 1):
        test_id = f"TC-{i:04d}"

        # 1. Build the structured payload (message = placeholder for text scenarios)
        payload = builder.build(scenario=scenario_filter)

        # 2. Annotate — Gemini fills in the message and description
        annotation = annotator.annotate(payload, message_mode=message_mode)
        payload["message"] = annotation["message"]
        payload["testDescription"] = annotation["testDescription"]
        payload["messageMode"] = annotation["messageMode"]
        if "guardrailCategory" in annotation:
            payload["guardrailCategory"] = annotation["guardrailCategory"]

        # 3. Save
        case_dir = output_dir / test_id
        case_dir.mkdir(parents=True, exist_ok=True)
        out_path = case_dir / "payload.json"
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved_paths.append(out_path)

        if verbose:
            scenario = payload.get("scenario", "")
            seg = payload.get("customer", {}).get("customerSegment", "?")
            mode = annotation["messageMode"]
            guardrail = annotation.get("guardrailCategory", "")
            mode_tag = f"{mode}/{guardrail}" if guardrail else mode
            msg_preview = payload["message"][:65].replace("\n", " ")
            print(
                f"  [{test_id}] {scenario:<20} seg={seg:<7} mode={mode_tag:<22} "
                f'desc="{annotation["testDescription"][:45]}"'
            )
            print(f"           message: {msg_preview!r}")

        # Small delay to stay within Gemini rate limits
        if annotator.available and i < n:
            time.sleep(0.5)

    if verbose:
        # Summary table
        from collections import Counter
        modes = Counter(json.loads(p.read_text())["messageMode"] for p in saved_paths)
        print(f"\nDone. {len(saved_paths)} file(s) written to {output_dir.resolve()}")
        print("Mode distribution:", dict(modes))

    return saved_paths


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random n8n orchestration payload test cases."
    )
    parser.add_argument(
        "n",
        type=int,
        help="Number of test cases to generate",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Gemini API key (default: reads GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        default=False,
        help="Skip Gemini; use static fallback messages instead",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT),
        help=f"Root folder for output (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default=None,
        help="Only generate a specific scenario type",
    )
    parser.add_argument(
        "--message-mode",
        choices=MESSAGE_MODES,
        default=None,
        help=(
            "Force a specific message mode for all cases. "
            "Default: random mix (strict 50%% / creative 30%% / adversarial 20%%)."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress progress output",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")

    generate(
        n=args.n,
        api_key=api_key,
        output_dir=Path(args.output_dir),
        use_ai=not args.no_ai,
        scenario_filter=args.scenario,
        message_mode=args.message_mode,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    _cli()
