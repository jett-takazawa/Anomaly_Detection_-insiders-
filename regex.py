# save as earnings_flags.py
import re
import pandas as pd
from datetime import datetime

# STRICT earnings title format:
# "Will {Company Name} ({TICKER}) beat quarterly earnings?"
EARNINGS_STRICT_RE = re.compile(
    r'^\s*will\s+(.+?)\s*\(([A-Z]{1,5})\)\s+beat\s+quarterly\s+earnings\?\s*$',
    re.IGNORECASE
)

# Optional: crypto PRICE/ACTION markets (not speech mentions)
COIN_RE = re.compile(
    r'\b(bitcoin|btc|ethereum|eth|xrp|solana|sol|doge|dogecoin|ada|cardano|ltc|litecoin|bch|trx|tron|usdt|tether|usdc|bnb|binance|dot|polkadot|avax|avalanche|xmr|monero|ripple)\b',
    re.IGNORECASE
)
# Expanded to catch ALL price movement keywords
MOVE_RE = re.compile(
    r'\b(up or down|above|below|price|ath|close|open|settle|futures|inflow|outflow|%|'
    r'reach|hit|cross|break|surge|drop|fall|rise|gain|lose|crash|moon|pump|dump|'
    r'target|rally|dip|spike|climb|plunge|soar|tank|rocket|bottom|peak|high|low|'
    r'by\b|\$\d+|\b\d{1,2}(:\d{2})?\s?(am|pm)?\s?(et|utc)\b)',
    re.IGNORECASE
)
SAY_MENTION_RE = re.compile(r'\b(say|mention|tweet|post|utter|state)\b', re.IGNORECASE)

def split_titles(s: str) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    # Split on |, strip spaces and smart quotes/parens
    parts = [p.strip(" \t\n\r\u201c\u201d\"'()") for p in s.split("|")]
    # De-dupe while preserving order
    seen, out = set(), []
    for p in parts:
        key = p.lower()
        if p and key not in seen:
            seen.add(key)
            out.append(p)
    return out

def is_earnings_strict(title: str):
    m = EARNINGS_STRICT_RE.match(title)
    if not m:
        return False, None, None
    company, ticker = m.group(1).strip(), m.group(2).upper()
    return True, company, ticker

def is_crypto_price_market(title: str) -> bool:
    # Require coin + movement context; exclude speech/mention prompts
    t = title
    return bool(COIN_RE.search(t) and MOVE_RE.search(t) and not SAY_MENTION_RE.search(t))

def validate_row(row) -> list[str]:
    """Validate row data and return list of validation errors."""
    errors = []

    # Validate wallet address exists
    wallet = row.get("wallet", "")
    if pd.isna(wallet) or not str(wallet).strip():
        errors.append("Missing wallet address")

    # Validate dates
    first_trade_ts = row.get("first_trade_ts", "")
    last_trade_ts = row.get("last_trade_ts", "")

    if pd.notna(first_trade_ts) and first_trade_ts and pd.notna(last_trade_ts) and last_trade_ts:
        try:
            first_dt = datetime.fromisoformat(str(first_trade_ts).replace('Z', '+00:00'))
            last_dt = datetime.fromisoformat(str(last_trade_ts).replace('Z', '+00:00'))

            # Sanity check: first_trade should be before last_trade
            if first_dt > last_dt:
                errors.append(f"first_trade_ts ({first_trade_ts}) is AFTER last_trade_ts ({last_trade_ts})")

            # Sanity check: trades shouldn't be in the future
            now = datetime.now(first_dt.tzinfo) if first_dt.tzinfo else datetime.now()
            if first_dt > now:
                errors.append(f"first_trade_ts ({first_trade_ts}) is in the FUTURE")
            if last_dt > now:
                errors.append(f"last_trade_ts ({last_trade_ts}) is in the FUTURE")

        except (ValueError, AttributeError) as e:
            errors.append(f"Invalid date format: {e}")

    # Validate that we have at least some market data
    active = row.get("active_market_titles", "")
    hist = row.get("historical_market_titles", "")
    if (pd.isna(active) or not str(active).strip()) and (pd.isna(hist) or not str(hist).strip()):
        errors.append("No active or historical market titles")

    # Validate numeric fields are reasonable
    trades_count = row.get("trades_count", 0)
    if pd.notna(trades_count) and trades_count < 0:
        errors.append(f"Negative trades_count: {trades_count}")

    positions_count = row.get("positions_count", 0)
    if pd.notna(positions_count) and positions_count < 0:
        errors.append(f"Negative positions_count: {positions_count}")

    return errors

def process_row(row):
    # VALIDATION FIRST - catch bad data early
    validation_errors = validate_row(row)

    # Normalize empty values to empty strings
    active = row.get("active_market_titles", "")
    hist   = row.get("historical_market_titles", "")
    # Handle NaN values from pandas (empty cells become float NaN)
    if pd.isna(active):
        active = ""
    else:
        active = str(active).strip()
    if pd.isna(hist):
        hist = ""
    else:
        hist = str(hist).strip()

    # Combine and split titles
    titles = split_titles((active or "") + (" | " if active and hist else "") + (hist or ""))

    earnings_count = 0
    earnings_tickers = set()

    for t in titles:
        ok, company, ticker = is_earnings_strict(t)
        if ok:
            earnings_count += 1
            earnings_tickers.add(ticker)

    # Simple binaries you asked for:
    other_earnings_markets = 1 if len(earnings_tickers) >= 2 else 0
    traded_crypto          = 1 if any(is_crypto_price_market(t) for t in titles) else 0

    # Calculate days since first trade with proper validation
    first_trade_ts = row.get("first_trade_ts", "")
    days_since_first_trade = 0
    if pd.notna(first_trade_ts) and first_trade_ts:
        try:
            # Parse ISO timestamp to datetime
            first_trade_dt = datetime.fromisoformat(str(first_trade_ts).replace('Z', '+00:00'))
            today_dt = datetime.now(first_trade_dt.tzinfo) if first_trade_dt.tzinfo else datetime.now()
            days_diff = (today_dt - first_trade_dt).days

            # Sanity check: days should be non-negative
            if days_diff < 0:
                validation_errors.append(f"first_trade_ts is in the future (days_diff={days_diff})")
                days_since_first_trade = 0
            else:
                days_since_first_trade = days_diff
        except (ValueError, AttributeError) as e:
            validation_errors.append(f"Error parsing first_trade_ts: {e}")
            days_since_first_trade = 0

    # Count total unique markets
    total_markets_count = len(titles)

    return pd.Series({
        "earnings_count": earnings_count,
        "earnings_distinct_issuers": len(earnings_tickers),
        "earnings_tickers_list": ",".join(sorted(earnings_tickers)) if earnings_tickers else "",
        "other_earnings_markets": other_earnings_markets,
        "traded_crypto": traded_crypto,
        "days_since_first_trade": days_since_first_trade,
        "total_markets_count": total_markets_count,
        "validation_errors": " | ".join(validation_errors) if validation_errors else ""
    })

if __name__ == "__main__":
    # Change 'input.csv' to your file path
    df = pd.read_csv("outputs/users_enriched_with_financials.csv")
    print(f"Processing {len(df)} rows...")

    features = df.apply(process_row, axis=1)
    out = pd.concat([df, features], axis=1)

    # Print validation summary
    rows_with_errors = out[out["validation_errors"] != ""]
    print(f"\nValidation Summary:")
    print(f"  Total rows: {len(out)}")
    print(f"  Rows with errors: {len(rows_with_errors)}")
    print(f"  Rows clean: {len(out) - len(rows_with_errors)}")

    if len(rows_with_errors) > 0:
        print(f"\nFirst 10 rows with validation errors:")
        for idx, row in rows_with_errors.head(10).iterrows():
            wallet = str(row.get("wallet", ""))[:10]
            errors = row["validation_errors"]
            print(f"  Row {idx} (wallet {wallet}...): {errors}")

    out.to_csv("output_with_signals.csv", index=False)
    print(f"\nWrote output_with_signals.csv")
