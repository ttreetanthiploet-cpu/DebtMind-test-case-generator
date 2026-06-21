"""
unit_test_main.py — CLI entry point for unit test generation.

Generates unit test cases for individual n8n agents by transforming
orchestration payloads (from final_test_case_gen/output/) into the exact
webhook input format each agent expects, then calling Gemini to annotate
the expected output.

Usage
─────
    # Generate 5 tests for each agent (random mode mix)
    python3 -m final_test_case_gen.unit_test_main all 5

    # Generate tests for a specific agent
    python3 -m final_test_case_gen.unit_test_main classification 5
    python3 -m final_test_case_gen.unit_test_main advisor 5
    python3 -m final_test_case_gen.unit_test_main summary 5
    python3 -m final_test_case_gen.unit_test_main output_guardrail 5

    # Dry-run (no Gemini)
    python3 -m final_test_case_gen.unit_test_main all 5 --no-ai

    # Call the live Classification webhook to get real narratives for advisor tests
    python3 -m final_test_case_gen.unit_test_main advisor 5 --call-classification-api

    # Only generate clean replies for guardrail tests (no deliberate violations)
    python3 -m final_test_case_gen.unit_test_main output_guardrail 5 --no-violation-replies

Output
──────
    final_test_case_gen/unit_tests/
        classification/TC-0001/test.json
        advisor/TC-0001/test.json
        summary/TC-0001/test.json
        output_guardrail/TC-0001/test.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from final_test_case_gen.unit_test_generator import (
    _DEFAULT_SOURCE,
    _DEFAULT_UNIT_TESTS,
    generate_advisor_tests,
    generate_classification_tests,
    generate_guardrail_tests,
    generate_summary_tests,
)

_AGENTS = ("classification", "advisor", "summary", "output_guardrail", "all")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate unit test cases for individual n8n agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "agent",
        choices=_AGENTS,
        help="Which agent to generate tests for ('all' runs all four)",
    )
    parser.add_argument(
        "n",
        type=int,
        help="Number of test cases to generate per agent",
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
        help="Skip Gemini; use static fallback expected outputs",
    )
    parser.add_argument(
        "--source-dir",
        default=str(_DEFAULT_SOURCE),
        help=f"Folder with TC-NNNN/payload.json files (default: {_DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_UNIT_TESTS),
        help=f"Root output folder (default: {_DEFAULT_UNIT_TESTS})",
    )
    parser.add_argument(
        "--call-classification-api",
        action="store_true",
        default=False,
        help=(
            "For advisor tests: call the live Intent Classification webhook "
            "to obtain the real narrative before building the advisor input. "
            "Requires network access to the n8n instance."
        ),
    )
    parser.add_argument(
        "--no-webhook",
        action="store_true",
        default=False,
        help=(
            "Skip calling the live n8n webhooks; use Gemini annotation only "
            "for expectedOutput. Useful when the n8n instance is not reachable."
        ),
    )
    parser.add_argument(
        "--no-violation-replies",
        action="store_true",
        default=False,
        help=(
            "For output_guardrail tests: do NOT generate deliberately bad bot replies; "
            "generate clean replies for all test cases instead."
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
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    use_ai = not args.no_ai
    call_webhook = not args.no_webhook
    verbose = not args.quiet

    agents_to_run = (
        ["classification", "advisor", "summary", "output_guardrail"]
        if args.agent == "all"
        else [args.agent]
    )

    all_saved: dict[str, list[Path]] = {}

    for agent in agents_to_run:
        if agent == "classification":
            all_saved[agent] = generate_classification_tests(
                n=args.n,
                source_dir=source_dir,
                output_dir=output_dir / "classification",
                api_key=api_key,
                use_ai=use_ai,
                call_webhook=call_webhook,
                verbose=verbose,
            )

        elif agent == "advisor":
            all_saved[agent] = generate_advisor_tests(
                n=args.n,
                source_dir=source_dir,
                output_dir=output_dir / "advisor",
                api_key=api_key,
                use_ai=use_ai,
                call_classification_api=args.call_classification_api,
                call_webhook=call_webhook,
                verbose=verbose,
            )

        elif agent == "summary":
            all_saved[agent] = generate_summary_tests(
                n=args.n,
                source_dir=source_dir,
                output_dir=output_dir / "summary",
                api_key=api_key,
                use_ai=use_ai,
                call_webhook=call_webhook,
                verbose=verbose,
            )

        elif agent == "output_guardrail":
            all_saved[agent] = generate_guardrail_tests(
                n=args.n,
                source_dir=source_dir,
                output_dir=output_dir / "output_guardrail",
                api_key=api_key,
                use_ai=use_ai,
                test_violation_replies=not args.no_violation_replies,
                call_webhook=call_webhook,
                verbose=verbose,
            )

    if verbose:
        print("\n" + "─" * 60)
        print("Unit test generation complete")
        for agent, paths in all_saved.items():
            print(f"  {agent:<18} {len(paths)} file(s) written")
        print(f"Output root: {output_dir.resolve()}")


if __name__ == "__main__":
    _cli()
