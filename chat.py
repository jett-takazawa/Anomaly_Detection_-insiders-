"""
Analyze user portfolios with LLM (Grok) to extract numeric features for ML.

This script:
1. Reads an input CSV (default: outputs/specific_users_enriched.csv)
2. For each user, extracts their *_market_titles
3. Calls Grok with an inlined llm_judge_prompt to get numeric analysis
4. Outputs outputs/user_data_final.csv with all original data + LLM scores
"""

import asyncio
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

# --- Env / Config ---
load_dotenv(Path(__file__).parent / ".env")

XAI_API_KEY = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
if not XAI_API_KEY:
    raise RuntimeError("Missing XAI_API_KEY (or GROK_API_KEY) in environment.")

GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-latest")
GROK_URL = "https://api.x.ai/v1/chat/completions"
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

NEUTRAL_SCORES = [0.0, 0.5, 0.5, 0.0, 50.0]  # [conflict, randomness, focus, variant, insider]

LLM_JUDGE_PROMPT = r"""

YOU ARE A as a numeric judge over a single wallet’s market titles.  
**The model must return numbers only** — a fixed-order JSON array with five numeric values.

OUTPUT NUMBERS ONLY. Return a JSON array of five numbers in this exact order:

1. `conflict_penalty` ∈ [0,1] — penalty for conflicting insider theses (e.g., multiple **large** earnings bets across unrelated issuers in the same window).
2. `randomness_penalty` ∈ [0,1] — penalty for breadth across unrelated domains (sports, weather, politics, macro, crypto, equities, etc.) higher number of markets penalized.
3. `focus_boost` ∈ [0,1] —   Multiple mentions of possible company or field. i.e (NETFLIX mentioned multiple times) and small amount of markets with high insider likelyhood.
4. `variant_chain_density` ∈ [0,1] — fraction of titles that are near-duplicate variants of the same thesis (e.g., same issuer with varying dates or minor phrasing).
5. `market_score` ∈ [0,100] — overall judgment that the wallet is pursuing a narrow, potentially insider‑advantaged thesis (higher is more insider‑like).

Do **not** include any keys, labels, prose, or explanations in your response. Output **only** the JSON array of five numbers in the order above.

---

## When analyzing markets always keep in mind the "Insider" Context...
# This should heavily influence market_score

Rules:
1) Insider-eligible when the outcome/timing is controlled or knowable early by a small circle (company, agency, court, deal team, party leadership) or reliably leak-prone (earnings, FDA, M&A/offerings, product launch with a date window, congressional deadlines/whip counts, major legal rulings).
2) Usually NOT insider-eligible: lotteries, generic weather, most sports match outcomes, macro prints with audited embargo (CPI, NFP), long-horizon vague culture events, stock or crypto movement.
3) Borderline sports are eligible ONLY when injury/lineup/discipline info could leak before public AND horizon ≤ 72h.
5) DO NOT INVENT FACTS. Base the decision ONLY on the provided fields.
6) Output must be strict JSON (no prose) per the schema.



## Scoring Rules (apply consistently)

- **Conflicting insider data → `conflict_penalty` up.**  
  Multiple *large* earnings bets across unrelated issuers in the same time window indicate conflicting theses (less likely to be genuine insider use).
  

- **Domain breadth → `randomness_penalty` up.**  
  Titles that span many unrelated domains (sports + weather + macro + several unrelated equities/crypto) increase randomness.
    If a user has high amounts of "GAMBLING" type activity, crank up the randomness penalty
    more markets = higher domean breadth
  
- **Issuer/product focus → `focus_boost` up.**  
  Multiple titles tied to the same issuer/product family suggest a narrow, high‑conviction thesis.
  
- **Overall `market_score` (0..100):**  
  Increases with high focus and "insider" trade focus, and decreases with high randomness and high conflict. Keep internally consistent with the four components.
  WE do not want to false label accounts unless truly suspicous. 
  Very few titles is suspect...
  ***SUSPECT ACTIVITY IS SHOWN WITH A SINGLE OR FEW HIGH INSIDER PROBABILITY MARKETS***
  LARGE whales will show up as suspicious but do not flag these accounts with 500 open positions and MANY market trades. Insider traders are often time not full time polymarket users. 

---

## Required Output Format

Return **only** a JSON array with five numbers in this exact order:

```json
[conflict_penalty, randomness_penalty, focus_boost, insider_likelihood]
```

**Examples of valid outputs:**
```json
[0.25, 0.60, 0.70, 58]
```
```json
[0.00, 0.05, 0.95, 74]
```

---
SCORING REMINDERS:
1) Conflicting insider data (multiple large earnings bets across unrelated issuers) → conflict_penalty↑
2) Many random/unrelated domains → randomness_penalty↑
3) Multiple markets for the same issuer/product family → focus_boost↑
4) insider_likelihood rises with focus/variants and falls with randomness/conflict

EXAMPLES:
🔴 High 

- Polymarket US go-live; Gemini 3.0 by Nov 15/30; Monad airdrop; . (Small circles control timing; leak-prone.)

- Legal ruling: Supreme Court on Trump tariffs. (Tiny chamber knows early.)

🟠 Medium-High

- FOMC 25 bps cut (Dec 2025). (Committee decision; tight but day-of leaks possible.) 

🟡 Situational 

- Geopolitics: US–Venezuela engagement; Russia–Ukraine ceasefire. (Negotiation leaks sporadic.)

- Leadership turnover: Maduro 2025 / Xi 2026. (Inner circle; low yield until late leaks.)

- Elections/margins: Mamdani/Kast/LLA seats—only during count.

🔵 Low / skip

- Crypto price movements

- Sports winners (T1/KT Rolster) unless last-minute injury leak.

- Satoshi moves BTC (unknowable).

- Search-rank (“#1 searched”) (aggregate behavior).

- Far-horizon elections (JD Vance 2028).

- Nonsense (Jesus before GTA VI).

```

OUTPUT: Return ONLY a JSON array of **five numbers** in the specified order.
```

---

## Few‑Shot Examples (anchor behavior; keep compact)



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
[0.00, 0.0, .87, 93]
```

### Example C — Conflicting large earnings bets (higher conflict; moderate randomness)
# ANY ACCOUNT WITH TWO OR MORE EARNINGS TYPE TRADES MUST GET HIGH PENALTY
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
[1, 0.2, 0.0, 3]
```

### Example D — Many markets with low "insider" likelyhood, 
### Impossible to gain insider information on crypto movements 
# ANY crypto bet is labled as low insider likelyhood...
```
TITLES:
Bitcoin Up or Down - October 20, 11AM ET 
Bitcoin Up or Down - October 20, 12PM ET 
Will Trump pardon Changpeng Zhao in 2025
Ethereum Up or Down - October 20, 11AM ET 
Ethereum Up or Down - October 20, 10AM ET 
XRP Up or Down - October 20, 12PM ET 

DERIVED STATS:
- total_titles: 6

OUTPUT:
[0, .22, 0.4, 8]

```


### Example E — Many markets with low "insider" likelyhood, 
### All titles are related to "insider" NETFLIX data, the most obvious suspect insider trader of all time. 
Will “Nobody Wants This” be the top global Netflix show this week? (November 4 2025) 
Will “The Witcher: Season 4” be the top global Netflix show this week? (November 4 2025) 
Will “The Perfect Neighbor” be the top global Netflix movie this week? (October 28) 
Will Netflix (NFLX) beat quarterly earnings?
Will “The Woman in Cabin 10” be the top global Netflix movie this week?
Will “KPop Demon Hunters” be the top global Netflix movie this week?

DERIVED STATS:
- total_titles: 6

OUTPUT:
[0, 0, 1, 99]
```


```

### Example F — Many markets with low "insider" likelyhood, 
### True Randomness is shown WITH LARGE AMOUNTS OF MARKETS 
MegaETH market cap (FDV) >$4B one day after launch?
Will Barcelona win the 2025–26 La Liga?
Will the price of Bitcoin be above $108,000 on November 9? etc etc etc...
........

DERIVED STATS:
- total_titles: 25!!

OUTPUT:
[0.15, 82, 0, 2]
```

### Example G — Single market with high "insider" likelyhood 
# A new account with very little activity with a bet on a high "insider" likelyhood market is seen as suspicious
Will Meta beat Earnings?
Will the Government shutdown end by November 15?

DERIVED STATS:
- total_titles: 2

OUTPUT:
[0, 12, 0, 82]
"""

def build_user_prompt(wallet: str, titles: list[str], time_window: str = "Recent") -> str:
    """Build the user message for LLM judge."""
    if not titles:
        titles = ["(No active positions)"]

    titles_text = "\n".join(titles)

    # Simple domain classification
    domain_counts = {
        "earnings": sum(1 for t in titles if "eps" in t.lower() or "earnings" in t.lower()),
        "product":  sum(1 for t in titles if "release" in t.lower() or "launch" in t.lower()),
        "legal":    sum(1 for t in titles if "lawsuit" in t.lower() or "trial" in t.lower()),
        "sports":   sum(1 for t in titles if any(s in t.lower() for s in ["win", "score", "game", "match"])),
        "weather":  sum(1 for t in titles if "weather" in t.lower() or "hurricane" in t.lower()),
        "other": 0
    }
    domain_counts["other"] = max(0, len(titles) - sum(domain_counts.values()))

    # Extract potential issuers (very rough heuristic)
    issuer_counts: dict[str, int] = {}
    for title in titles:
        for word in title.split():
            if word.isupper() and 2 <= len(word) <= 5 and word.isalpha():
                issuer_counts[word] = issuer_counts.get(word, 0) + 1

    issuer_run_max = max(issuer_counts.values()) if issuer_counts else 0

    user_message = f"""WALLET: {wallet}
TIME WINDOW: {time_window}

TITLES (one per line):
{titles_text}

DERIVED STATS (for reference only; do not repeat in output):
- total_titles: {len(titles)}
- domain_counts: {json.dumps(domain_counts, ensure_ascii=False)}
- issuer_counts: {json.dumps(issuer_counts, ensure_ascii=False)}
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


def _extract_scores_array(text: str) -> list[float]:
    """
    Extract and validate a JSON array of five numbers from model text.
    Returns NEUTRAL_SCORES if invalid.
    """
    # Prefer explicit JSON array match
    m = re.search(r"\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*){4}-?\d+(?:\.\d+)?\s*\]", text)
    candidate = m.group(0) if m else text.strip()
    try:
        arr = json.loads(candidate)
        if isinstance(arr, list) and len(arr) == 5 and all(isinstance(x, (int, float)) for x in arr):
            # Clamp to bounds
            c, r, f, v, i = arr
            c = min(max(float(c), 0.0), 1.0)
            r = min(max(float(r), 0.0), 1.0)
            f = min(max(float(f), 0.0), 1.0)
            v = min(max(float(v), 0.0), 1.0)
            i = min(max(float(i), 0.0), 100.0)
            return [c, r, f, v, i]
    except Exception:
        pass
    return NEUTRAL_SCORES


async def call_grok(system_prompt: str, user_prompt: str) -> list[float]:
    """Call Grok chat.completions and return a 5-number score array."""
    payload = {
        "model": GROK_MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(GROK_URL, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return _extract_scores_array(content)


async def analyze_users(input_csv: Path, output_csv: Path):
    """Read enriched users CSV, analyze each with Grok, write final CSV."""
    print(f"Loading users from: {input_csv}")
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        users = list(reader)

    print(f"Found {len(users)} users to analyze\n")
    enriched_users = []

    for i, user in enumerate(users, 1):
        wallet = user.get("wallet", "(unknown)")
        print(f"[{i}/{len(users)}] Analyzing wallet: {wallet[:20]}...")

        # Prefer historical titles, else active
        historical_str = user.get("historical_market_titles", "") or ""
        active_str = user.get("active_market_titles", "") or ""

        if historical_str.strip():
            titles_str = historical_str
            source = "historical"
        elif active_str.strip():
            titles_str = active_str
            source = "active"
        else:
            titles_str = ""
            source = "none"

        titles = [t.strip() for t in titles_str.split("|") if t.strip()] if titles_str else []

        if not titles:
            print("  [INFO] No market data - using neutral scores")
            scores = NEUTRAL_SCORES
        else:
            print(f"  [INFO] Analyzing {len(titles)} {source} market titles...")
            user_prompt = build_user_prompt(wallet, titles)
            scores = await call_grok(LLM_JUDGE_PROMPT, user_prompt)
            print(f"  [SCORES] {scores}")

        enriched_user = dict(user)
        enriched_user["conflict_penalty"] = scores[0]
        enriched_user["randomness_penalty"] = scores[1]
        enriched_user["focus_boost"] = scores[2]
        enriched_user["variant_chain_density"] = scores[3]
        enriched_user["insider_likelihood"] = scores[4]

        # Calculate days_since_first_trade
        first_trade_ts = user.get("first_trade_ts", "")
        if first_trade_ts:
            try:
                # Parse ISO timestamp to datetime
                first_trade_dt = datetime.fromisoformat(first_trade_ts.replace('Z', '+00:00'))
                today_dt = datetime.now(first_trade_dt.tzinfo) if first_trade_dt.tzinfo else datetime.now()
                days_diff = (today_dt - first_trade_dt).days
                enriched_user["days_since_first_trade"] = days_diff
            except (ValueError, AttributeError):
                enriched_user["days_since_first_trade"] = 0
        else:
            enriched_user["days_since_first_trade"] = 0

        enriched_users.append(enriched_user)

    # Write output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting results to: {output_csv}")

    base_fields = list(users[0].keys()) if users else []
    score_fields = [
        "conflict_penalty",
        "randomness_penalty",
        "focus_boost",
        "variant_chain_density",
        "insider_likelihood",
        "days_since_first_trade",
    ]
    all_fields = base_fields + [f for f in score_fields if f not in base_fields]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(enriched_users)

    print(f"\n[OK] Analysis complete! {len(enriched_users)} users written to {output_csv}")

    # Summary
    print("\n=== Summary ===")
    for user in enriched_users:
        wl = user.get("wallet", "(unknown)")
        print(f"\nWallet: {wl[:20]}...")
        print(f"  Insider Likelihood: {float(user['insider_likelihood']):.1f}/100")
        print(f"  Focus Boost: {float(user['focus_boost']):.2f}")
        print(f"  Randomness Penalty: {float(user['randomness_penalty']):.2f}")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    # Default aligned with docstring; CLI arg overrides
    default_in = script_dir / "outputs" / "market_users_enriched.csv"
    input_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else default_in
    output_csv = script_dir / "outputs" / "user_data_final.csv"

    if not input_csv.exists():
        print(f"[ERROR] Input file not found: {input_csv}")
        sys.exit(1)

    asyncio.run(analyze_users(input_csv, output_csv))
