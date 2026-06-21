# final_test_case_gen

Generates random orchestration payload test cases for the n8n Prototype_v1.2 webhook.
Each output file is the exact JSON body the webapp POSTs to n8n, enriched with
`testDescription` and `scenario` fields.

---

## Setup

### 1. Install dependencies

```bash
pip3 install google-genai requests
```

### 2. Set your Gemini API key

The code reads `GEMINI_API_KEY` from the environment. Set it once per session:

```bash
export GEMINI_API_KEY=your_key_here
```

To persist it permanently, add the line above to `~/.zshrc` (or `~/.bashrc`) and run `source ~/.zshrc`.

You can also pass the key directly on the command line with `--api-key your_key_here` instead of exporting it.

> The code does **not** auto-load `.env.local`. If you use one, run `source .env.local` in your shell first.

---

## Running the scripts

There are two ways to run without any install step.

### Option A — run from inside the project folder (simplest)

```bash
cd /path/to/final_test_case_gen

# Generate orchestration payloads
python3 main.py 10

# Generate unit tests
python3 unit_test_main.py all 5
```

### Option B — run from anywhere using `PYTHONPATH`

```bash
export PYTHONPATH=/path/to/work   # parent folder of final_test_case_gen/

python3 -m final_test_case_gen.main 10
python3 -m final_test_case_gen.unit_test_main all 5
```

### Option C — install once for global CLI commands

```bash
pip3 install -e /path/to/final_test_case_gen/
```

This registers `gen-payloads` and `gen-unit-tests` commands that work from any directory without any path setup.

---

## Quick start

```bash
cd /path/to/final_test_case_gen
export GEMINI_API_KEY=your_key_here

# Step 1 — generate orchestration payloads
python3 main.py 10

# Step 2 — generate unit tests for all agents (reads from output/ created above)
python3 unit_test_main.py all 5
```

Dry-run (no API key needed):

```bash
python3 main.py 10 --no-ai
python3 unit_test_main.py all 5 --no-ai
```

---

## Orchestration payload generation (`main.py`)

```bash
# Random mode mix — strict 50% / creative 30% / adversarial 20% (default)
python3 main.py 10

# Force a specific message mode for all cases
python3 main.py 10 --message-mode strict
python3 main.py 10 --message-mode creative
python3 main.py 10 --message-mode adversarial

# Only one scenario type
python3 main.py 5 --scenario NEGOTIATE_PAYMENT

# Dry-run — skip Gemini, use static fallback messages
python3 main.py 10 --no-ai

# Custom output directory
python3 main.py 10 --output-dir ./my_cases

# Pass API key directly
python3 main.py 10 --api-key your_key_here
```

### Scenario types (weighted randomly)

| Scenario | Description |
|---|---|
| `NEW_CONVERSATION` | Fresh session, no history, no offers |
| `AFTER_OFFER_TEXT` | Offers already shown; customer replies via text |
| `OFFER_SELECTED` | Customer selects one offer (`messageType=JSON`) |
| `NEGOTIATE_PAYMENT` | Customer wants to lower installment after seeing offers |
| `NEGOTIATE_TERM` | Customer wants shorter/longer term after seeing offers |
| `STAFF_REQUEST` | Customer asks to be contacted by staff |
| `EDUCATION_QUESTION` | Customer asks a conceptual question about debt restructuring |
| `OFF_TOPIC` | Customer message unrelated to debt or banking |
| `MULTI_TURN` | Mid-conversation using real history slices from the database |

### Message modes

| Mode | Description |
|---|---|
| `strict` | Message strictly matches the planned scenario |
| `creative` | Gemini deviates freely — unexpected but still plausible |
| `adversarial` | Designed to stress-test the output guardrail |

Adversarial cases carry a `guardrailCategory` field:
`jailbreak`, `data_fishing`, `nsfw_abusive`, `hallucination_bait`, `prompt_injection`, `social_engineering`.

### Output

```
final_test_case_gen/output/
  TC-0001/payload.json
  TC-0002/payload.json
  ...
```

Each `payload.json` contains the full n8n orchestration payload plus:

- `"testDescription"` — Gemini-generated description with mode tag `[strict]`, `[creative]`, or `[adversarial/category]`
- `"scenario"` — scenario type used
- `"messageMode"` — `strict` | `creative` | `adversarial`
- `"guardrailCategory"` — *(adversarial only)* which attack type was used

---

## Unit test generation (`unit_test_main.py`)

Transforms orchestration payloads (from `output/`) into each agent's exact webhook input
format and uses Gemini to annotate the expected output.

```bash
# Generate 5 unit tests for every agent
python3 unit_test_main.py all 5

# One agent at a time
python3 unit_test_main.py classification 10
python3 unit_test_main.py advisor 10
python3 unit_test_main.py summary 10
python3 unit_test_main.py output_guardrail 10

# Dry-run — no Gemini, static fallback expected outputs
python3 unit_test_main.py all 5 --no-ai

# Call the live Classification webhook for real narratives (advisor tests only)
python3 unit_test_main.py advisor 5 --call-classification-api

# Always generate clean bot replies for guardrail tests (no deliberate violations)
python3 unit_test_main.py output_guardrail 5 --no-violation-replies

# Custom source / output directories
python3 unit_test_main.py all 5 --source-dir ./my_payloads --output-dir ./my_unit_tests

# Pass API key directly
python3 unit_test_main.py all 5 --api-key your_key_here
```

### What each agent generator does

| Agent | Source filter | Webhook | Expected output predicted by Gemini |
|---|---|---|---|
| `classification` | TEXT messages only | `dbce5b9e` | `route_to`, `narrative` |
| `advisor` | TEXT, non-offer-selected | `b7607735` | `consultAcc`, `maxPayment`, `maxTerm`, `DebtSituation`, `reConfirmMessage`, … |
| `summary` | All — auto-detects offer-selected vs staff-contact path | `515736a7` | Thai staff memo: `subject`, `objective`, `debt_cause`, `offer_suitability`, `request_information`, `summary` |
| `output_guardrail` | All | `efab08a3` | `fail_outputGuardrail`, `preCheckViolations`, `personalData`, `nsfw`, `hallucinationHarm` |

### Output

```
final_test_case_gen/unit_tests/
  classification/TC-0001/test.json
  advisor/TC-0001/test.json
  summary/TC-0001/test.json
  output_guardrail/TC-0001/test.json
```

Each `test.json`:

```jsonc
{
  "testId":            "TC-0001",
  "agentType":         "classification",
  "sourcePayload":     "TC-0007",
  "scenario":          "STAFF_REQUEST",
  "messageMode":       "adversarial",
  "guardrailCategory": "social_engineering",  // null for non-adversarial
  "testDescription":   "[adversarial/social_engineering] STAFF_REQUEST — expected route_to=summary",
  "webhookPath":       "dbce5b9e-1397-459a-871a-5b27433f1640",
  "input":             { /* ready-to-POST webhook payload */ },
  "expectedOutput":    { /* Gemini-annotated expected response */ }
}
```

Extra fields per agent:
- **summary**: `"summaryPath": "offer_selected" | "staff_contact"`
- **output_guardrail**: `"testSubtype": "violation_test" | "clean_test"`, `"isViolationTest": true | false`

---

## How it works

### Orchestration payload generation

| Step | Component | Gemini? |
|---|---|---|
| Sample customer, accounts, session, offers from DB | `db_sampler.py` | No |
| Pick scenario, build payload structure | `payload_builder.py` | No |
| Fill in user `message` + `testDescription` | `gemini_annotator.py` | **Yes — 1 call** |

### Unit test generation — Gemini calls per agent

| Agent | Gemini calls | Live webhook |
|---|---|---|
| `classification` | 1 — predicts `route_to` + `narrative` | Never |
| `advisor` | 1 — predicts extractor output fields | Only with `--call-classification-api` |
| `summary` | 1 — predicts full Thai staff memo | Never |
| `output_guardrail` | 2 — generates bot reply, then predicts guardrail verdict | Never |

Without `GEMINI_API_KEY` or with `--no-ai`, all generators fall back to static heuristics
(route from scenario name, canned Thai memo sentences, keyword-based guardrail verdict).

---

## Source files

| File | Purpose |
|---|---|
| `db_sampler.py` | Loads and indexes all database tables; typed helpers to sample customers, accounts, sessions, offers |
| `payload_builder.py` | Builds random orchestration payloads across 9 scenario types using real DB data |
| `gemini_annotator.py` | Calls `gemini-2.5-flash` to generate Thai user message + English test description |
| `main.py` | CLI entry point for orchestration payload generation |
| `transform.py` | Transforms orchestration payloads into per-agent webhook input formats |
| `unit_test_generator.py` | Four generator functions — one per agent |
| `unit_test_main.py` | CLI entry point for unit test generation |
