"""
Analyze user portfolios with LLM to extract numeric features for ML.

This script:
1. Reads specific_users_enriched.csv
2. For each user, extracts their active_market_titles
3. Calls an LLM with the llm_judge_prompt.md to get numeric analysis
4. Outputs user_data_final.csv with all original data + LLM scores
"""

import asyncio
import csv
import json
import os
import re
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).parent / ".env")

# LLM configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
XAI_API_KEY = os.getenv("XAI_API_KEY")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

# API endpoints
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
XAI_URL = "https://api.x.ai/v1/chat/completions"


LLM_JUDGE_PROMPT = """

# LLM Wallet Title Scoring Prompt (Numbers-Only)

This prompt makes an LLM act as a numeric judge over a single wallet’s market titles.  
**The model must return numbers only** — a fixed-order JSON array with five numeric values.

---

## System Message

You output numbers only. Return a JSON array of five numbers in this exact order:

1. `conflict_penalty` ∈ [0,1] — penalty for conflicting insider theses (e.g., multiple **large** earnings bets across unrelated issuers in the same window).
2. `randomness_penalty` ∈ [0,1] — penalty for breadth across unrelated domains (sports, weather, politics, macro, crypto, equities, etc.).
3. `focus_boost` ∈ [0,1] — boost for tight thematic focus (many markets tied to the same issuer/product family).
4. `variant_chain_density` ∈ [0,1] — fraction of titles that are near-duplicate variants of the same thesis (e.g., same issuer with varying dates or minor phrasing).
5. `market_score` ∈ [0,100] — overall judgment that the wallet is pursuing a narrow, potentially insider‑advantaged thesis (higher is more insider‑like).

Do **not** include any keys, labels, prose, or explanations in your response. Output **only** the JSON array of five numbers in the order above.

---

## Scoring Rules (apply consistently)

- **Conflicting insider data → `conflict_penalty` up.**  
  Multiple *large* earnings bets across unrelated issuers in the same time window indicate conflicting theses (less likely to be genuine insider use).

- **Domain breadth → `randomness_penalty` up.**  
  Titles that span many unrelated domains (sports + weather + macro + several unrelated equities/crypto) increase randomness.

- **Issuer/product focus → `focus_boost` up.**  
  Multiple titles tied to the same issuer/product family suggest a narrow, high‑conviction thesis.

- **Variant chain → `variant_chain_density` up.**  
  Repetitive variants of the same thesis (e.g., “Will Gemini 3.0 be released by Nov 15/22/30?”) increase density.

- **Overall `market_score` (0..100):**  
  Increases with high focus and variant density, and decreases with high randomness and high conflict. Keep internally consistent with the four components.

---

## Required Output Format

Return **only** a JSON array with five numbers in this exact order:

```json
[conflict_penalty, randomness_penalty, focus_boost, variant_chain_density, insider_likelihood]
```

**Examples of valid outputs:**
```json
[0.25, 0.60, 0.70, 0.40, 58]
```
```json
[0.00, 0.05, 0.95, 0.90, 74]
```

---
SCORING REMINDERS:
1) Conflicting insider data (multiple large earnings bets across unrelated issuers) → conflict_penalty↑
2) Many random/unrelated domains → randomness_penalty↑
3) Multiple markets for the same issuer/product family → focus_boost↑
4) Repetitive variants of the same thesis → variant_chain_density↑
5) insider_likelihood rises with focus/variants and falls with randomness/conflict

OUTPUT: Return ONLY a JSON array of **five numbers** in the specified order.
```

---

## Few‑Shot Examples (anchor behavior; keep compact)

### Example A — Scattered across domains (high randomness, low focus)
```
TITLES:
Will a hurricane make U.S. landfall by Nov 30?
Will the Lakers win on Friday?
Will NVDA close above $150 on Nov 15?

DERIVED STATS:
- total_titles: 3
- domain_counts: {"weather":1,"sports":1,"other":1}
- issuer_counts: {"NVDA":1,"GENERIC":2}
- issuer_run_max: 1
- earnings_titles: 0 across 0 issuers
- large_positions: []

OUTPUT:
[0.00, 0.80, 0.10, 0.10, 18]
```

### Example B — Tight product focus (low randomness/conflict, high focus & variants)
```
TITLES:
Will Gemini 3.0 be released by Nov 15?
Will Gemini 3.0 be released by Nov 22?
Will Gemini 3.0 be released by Nov 30?

DERIVED STATS:
- total_titles: 3
- domain_counts: {"product":3}
- issuer_counts: {"GOOGLE":3}
- issuer_run_max: 3
- earnings_titles: 0 across 0 issuers
- large_positions: []

OUTPUT:
[0.00, 0.05, 0.95, 0.90, 74]
```

### Example C — Conflicting large earnings bets (higher conflict; moderate randomness)
```
TITLES:
Will TSLA beat quarterly EPS in Q4?
Will AAPL beat quarterly EPS this week?
Will MSFT beat EPS on next report?

DERIVED STATS:
- total_titles: 3
- domain_counts: {"earnings":3}
- issuer_counts: {"TSLA":1,"AAPL":1,"MSFT":1}
- issuer_run_max: 1
- earnings_titles: 3 across 3 issuers
- large_positions: [{"issuer":"TSLA","size":"large"},{"issuer":"AAPL","size":"large"},{"issuer":"MSFT","size":"large"}]

OUTPUT:
[0.80, 0.20, 0.25, 0.10, 32]
```

---

## Validation Hints (caller-side, not seen by the model)

- Ensure the response parses as a JSON array of length 5.
- Bounds:
  - `conflict_penalty`, `randomness_penalty`, `focus_boost`, `variant_chain_density` ∈ [0,1]
  - `insider_likelihood` ∈ [0,100]
- If invalid, **retry once** with the original prompt. If still invalid, fallback to neutral:
  `[0.0, 0.5, 0.5, 0.0, 50]`.



"""



def build_user_prompt(wallet: str, titles: list[str], time_window: str = "Recent") -> str:
    """
    Build the user message for LLM judge.

    Args:
        wallet: Wallet address
        titles: List of market titles
        time_window: Time window description

    Returns:
        Formatted user prompt
    """
    if not titles:
        titles = ["(No active positions)"]

    titles_text = "\n".join(titles)

    # Simple domain classification
    domain_counts = {
        "earnings": sum(1 for t in titles if "EPS" in t or "earnings" in t.lower()),
        "product": sum(1 for t in titles if "release" in t.lower() or "launch" in t.lower()),
        "legal": sum(1 for t in titles if "lawsuit" in t.lower() or "trial" in t.lower()),
        "sports": sum(1 for t in titles if any(s in t.lower() for s in ["win", "score", "game", "match"])),
        "weather": sum(1 for t in titles if "weather" in t.lower() or "hurricane" in t.lower()),
        "other": 0
    }
    domain_counts["other"] = len(titles) - sum(domain_counts.values())

    # Extract potential issuers (simplified)
    issuer_counts = {}
    for title in titles:
        # Look for ticker-like patterns or company names
        words = title.split()
        for word in words:
            if word.isupper() and 2 <= len(word) <= 5:
                issuer_counts[word] = issuer_counts.get(word, 0) + 1

    issuer_run_max = max(issuer_counts.values()) if issuer_counts else 0


## User Message Template (fill placeholders)


    user_message = f"""WALLET: {wallet}
TIME WINDOW: {time_window}

TITLES (one per line):
{titles_text}

DERIVED STATS (for reference only; do not repeat in output):
- total_titles: {len(titles)}
- domain_counts: {json.dumps(domain_counts)}
- issuer_counts: {json.dumps(issuer_counts) if issuer_counts else "{}"}
- issuer_run_max: {issuer_run_max}
- earnings_titles: {domain_counts["earnings"]} across {len(set(issuer_counts.keys()))} issuers
- large_positions: []

SCORING REMINDERS:
1) Conflicting insider data (multiple large earnings bets across unrelated issuers) → conflict_penalty↑
2) Many random/unrelated domains → randomness_penalty↑
3) Multiple markets for the same issuer/product family → focus_boost↑
4) Repetitive variants of the same thesis → variant_chain_density↑
5) insider_likelihood rises with focus/variants and falls with randomness/conflict

OUTPUT: Return ONLY a JSON array of **five numbers** in the specified order.
"""
    return user_message


async def call_llm(system_prompt: str, user_prompt: str) -> list[float]:
    """
    Call LLM API and parse numeric response.

    Returns:
        List of 5 numbers: [conflict_penalty, randomness_penalty, focus_boost,
                            variant_chain_density, insider_likelihood]
    """
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")
        url = OPENAI_URL
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        model = OPENAI_MODEL
    elif LLM_PROVIDER == "xai":
        if not XAI_API_KEY:
            raise ValueError("XAI_API_KEY not set")
        url = XAI_URL
        headers = {"Authorization": f"Bearer {XAI_API_KEY}"}
        model = "grok-beta"
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")
    
    payload = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},],
        max_tokens= 500,
        temperature=TEMPERATURE,
        seed=42,
    )

    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(2):  # Retry once if needed
            try:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()

                # Try to extract JSON array from response (may have markdown or text around it)
                # Look for [...] pattern
                json_match = re.search(r'\[[\d\s,\.]+\]', content)
                if json_match:
                    content = json_match.group(0)

                # Parse JSON array
                scores = json.loads(content)

                # Validate
                if isinstance(scores, list) and len(scores) == 5:
                    # Ensure proper types and bounds
                    conflict = float(scores[0])
                    randomness = float(scores[1])
                    focus = float(scores[2])
                    variant = float(scores[3])
                    likelihood = float(scores[4])

                    # Validate bounds
                    assert 0 <= conflict <= 1, "conflict_penalty out of bounds"
                    assert 0 <= randomness <= 1, "randomness_penalty out of bounds"
                    assert 0 <= focus <= 1, "focus_boost out of bounds"
                    assert 0 <= variant <= 1, "variant_chain_density out of bounds"
                    assert 0 <= likelihood <= 100, "insider_likelihood out of bounds"

                    return [conflict, randomness, focus, variant, likelihood]
                else:
                    raise ValueError(f"Invalid response format: {content}")

            except Exception as e:
                print(f"  [WARNING] Attempt {attempt + 1} failed: {e}")
                if attempt == 1:  # Last attempt
                    print("  [FALLBACK] Using neutral scores")
                    return [0.0, 0.5, 0.5, 0.0, 50.0]
                await asyncio.sleep(2)

    return [0.0, 0.5, 0.5, 0.0, 50.0]  # Fallback


async def analyze_users(input_csv: Path, output_csv: Path):
    """
    Read enriched users CSV, analyze each with LLM, write final CSV.
    """
    print(f"Loading users from: {input_csv}")

    # Read input CSV
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        users = list(reader)

    print(f"Found {len(users)} users to analyze\n")


    # Analyze each user
    enriched_users = []

    for i, user in enumerate(users, 1):
        wallet = user["wallet"]
        print(f"[{i}/{len(users)}] Analyzing wallet: {wallet[:20]}...")

        # Parse market titles - prefer historical titles over active
        historical_str = user.get("historical_market_titles", "")
        active_str = user.get("active_market_titles", "")

        # Use historical if available, otherwise active
        if historical_str:
            titles_str = historical_str
            source = "historical"
        elif active_str:
            titles_str = active_str
            source = "active"
        else:
            titles_str = ""
            source = "none"

        titles = [t.strip() for t in titles_str.split("|") if t.strip()] if titles_str else []

        if not titles:
            print("  [INFO] No market data - using neutral scores")
            scores = [0.0, 0.5, 0.5, 0.0, 50.0]
        else:
            print(f"  [INFO] Analyzing {len(titles)} {source} market titles...")

            # Build prompt
            user_prompt = build_user_prompt(wallet, titles)

            # Call LLM
            scores = await call_llm(LLM_JUDGE_PROMPT, user_prompt)
            print(f"  [SCORES] {scores}")

        # Combine original data with LLM scores
        enriched_user = {**user}
        enriched_user["conflict_penalty"] = scores[0]
        enriched_user["randomness_penalty"] = scores[1]
        enriched_user["focus_boost"] = scores[2]
        enriched_user["variant_chain_density"] = scores[3]
        enriched_user["insider_likelihood"] = scores[4]

        enriched_users.append(enriched_user)

    # Write output CSV
    print(f"\nWriting results to: {output_csv}")

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        # Define field order
        base_fields = list(users[0].keys())
        score_fields = [
            "conflict_penalty",
            "randomness_penalty",
            "focus_boost",
            "variant_chain_density",
            "insider_likelihood",
        ]
        all_fields = base_fields + score_fields

        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(enriched_users)

    print(f"\n[OK] Analysis complete! {len(enriched_users)} users written to {output_csv}")

    # Print summary
    print("\n=== Summary ===")
    for user in enriched_users:
        print(f"\nWallet: {user['wallet'][:20]}...")
        print(f"  Insider Likelihood: {user['insider_likelihood']:.1f}/100")
        print(f"  Focus Boost: {user['focus_boost']:.2f}")
        print(f"  Randomness Penalty: {user['randomness_penalty']:.2f}")


if __name__ == "__main__":
    # Paths
    script_dir = Path(__file__).parent
    input_csv = script_dir / "outputs" / "specific_users_enriched.csv"
    output_csv = script_dir / "outputs" / "user_data_final.csv"

    # Allow custom input file
    if len(sys.argv) > 1:
        input_csv = Path(sys.argv[1])

    if not input_csv.exists():
        print(f"[ERROR] Input file not found: {input_csv}")
        sys.exit(1)

    asyncio.run(analyze_users(input_csv, output_csv))
