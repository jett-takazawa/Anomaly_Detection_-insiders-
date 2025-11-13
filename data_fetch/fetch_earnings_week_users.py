"""
Fetch and enrich users from specific earnings markets for the week.

Reads market URLs from earnings_markets_nov_11_16 file, fetches users,
enriches them with API data, and adds financial metrics.
"""

import asyncio
import csv
import logging
from pathlib import Path

import httpx

# Import existing functions
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_fetch.fetch_user_data import (
    fetch_users_from_market,
    enrich_user,
)
from data_fetch.utils import dedupe_by_key, write_csv_rows
from add_financial_metrics import enrich_user_financials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"


async def search_market_by_slug(slug: str, all_markets: list[dict]) -> str | None:
    """
    Find market ID by searching through market list for matching slug.

    Args:
        slug: Market slug (e.g., "oklo-quarterly-earnings-gaap-eps-11-11-2025-neg0pt12")
        all_markets: List of all markets from Gamma API

    Returns:
        Market condition ID or None if not found
    """
    # Normalize slug for comparison
    slug_normalized = slug.lower().strip()

    for market in all_markets:
        market_slug = (market.get("slug") or "").lower().strip()

        if market_slug == slug_normalized:
            market_id = market.get("conditionId") or market.get("condition_id") or market.get("id")
            if market_id:
                logger.info(f"Found market ID for {slug}: {market_id}")
                return str(market_id)

    logger.warning(f"No market found for slug: {slug}")
    return None


async def fetch_all_markets(client: httpx.AsyncClient, limit: int = 1000) -> list[dict]:
    """
    Fetch all open markets from Gamma API.

    Args:
        client: HTTP client
        limit: Maximum markets to fetch

    Returns:
        List of market dicts
    """
    logger.info(f"Fetching up to {limit} markets from Gamma API...")

    all_markets = []
    offset = 0
    per_page = 100

    while len(all_markets) < limit:
        remaining = limit - len(all_markets)
        fetch_size = min(per_page, remaining)

        url = f"{GAMMA_BASE_URL}/markets"
        params = {
            "limit": fetch_size,
            "offset": offset,
            "closed": "false",  # Open markets
            "order": "volume",
            "ascending": "false",
        }

        try:
            response = await client.get(url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            items = data if isinstance(data, list) else data.get("data", [])

            if not items:
                logger.info(f"No more markets available (stopped at {len(all_markets)} markets)")
                break

            all_markets.extend(items)
            logger.info(f"Fetched {len(items)} markets (total: {len(all_markets)})")

            if len(items) < fetch_size:
                logger.info(f"Reached end of available markets at {len(all_markets)}")
                break

            offset += fetch_size
            await asyncio.sleep(0.3)

        except Exception as e:
            logger.error(f"Error fetching markets at offset {offset}: {e}")
            break

    logger.info(f"Successfully fetched {len(all_markets)} total markets")
    return all_markets


def parse_market_urls(file_path: Path) -> list[str]:
    """
    Parse market URLs from markdown file and extract slugs.

    Args:
        file_path: Path to markdown file with URLs

    Returns:
        List of market slugs
    """
    logger.info(f"Reading market URLs from {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    slugs = []
    for line in lines:
        line = line.strip()
        if line.startswith("https://polymarket.com/event/"):
            # Extract slug from URL: https://polymarket.com/event/SLUG?tid=...
            slug = line.split("/event/")[1].split("?")[0]
            slugs.append(slug)

    logger.info(f"Found {len(slugs)} market URLs")
    return slugs


async def fetch_all_users_from_markets(market_ids: list[str]) -> list[dict]:
    """
    Fetch all users from a list of market IDs.

    Args:
        market_ids: List of market condition IDs

    Returns:
        List of user dicts with wallet, username, market_id
    """
    logger.info(f"Fetching users from {len(market_ids)} markets")

    all_users = []

    for i, market_id in enumerate(market_ids, 1):
        logger.info(f"Processing market {i}/{len(market_ids)}: {market_id}")

        try:
            users = await fetch_users_from_market(market_id)
            all_users.extend(users)
            logger.info(f"Found {len(users)} users in this market (total: {len(all_users)})")
        except Exception as e:
            logger.error(f"Failed to fetch users from market {market_id}: {e}")
            continue

        # Rate limiting delay
        if i < len(market_ids):
            await asyncio.sleep(0.5)

    logger.info(f"Fetched {len(all_users)} total user records")
    return all_users


async def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).parent.parent
    markets_file = project_root / "earnings_markets_nov_11_16"
    output_dir = Path(__file__).parent / "outputs_earnings_week"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Earnings Week User Data Collection ===")
    logger.info(f"Markets file: {markets_file}")
    logger.info(f"Output directory: {output_dir}")

    # Step 1: Parse market URLs
    if not markets_file.exists():
        logger.error(f"Markets file not found: {markets_file}")
        return

    market_slugs = parse_market_urls(markets_file)

    if not market_slugs:
        logger.error("No market URLs found in file")
        return

    # Step 2: Fetch all open markets from Gamma API
    logger.info(f"\n[Step 1/4] Fetching all open markets from Gamma API...")

    async with httpx.AsyncClient() as client:
        all_markets = await fetch_all_markets(client, limit=25000)

    logger.info(f"Fetched {len(all_markets)} total markets")

    # Step 3: Search for market IDs matching our slugs
    logger.info(f"\n[Step 2/4] Searching for {len(market_slugs)} earnings markets...")
    market_ids = []

    for i, slug in enumerate(market_slugs, 1):
        logger.info(f"Searching for market {i}/{len(market_slugs)}: {slug}")
        market_id = await search_market_by_slug(slug, all_markets)

        if market_id:
            market_ids.append(market_id)

    logger.info(f"Successfully resolved {len(market_ids)} market IDs")

    if not market_ids:
        logger.error("No market IDs found - cannot proceed")
        return

    # Step 4: Fetch all users from these markets
    logger.info(f"\n[Step 3/5] Fetching users from {len(market_ids)} markets...")
    all_users = await fetch_all_users_from_markets(market_ids)

    if not all_users:
        logger.error("No users found in any market")
        return

    # Deduplicate users by wallet
    unique_users = dedupe_by_key(all_users, "wallet")
    logger.info(f"Deduplicated to {len(unique_users)} unique users")

    # Write raw users CSV
    users_raw_path = output_dir / "users_raw.csv"
    raw_rows = [
        {
            "wallet": u["wallet"],
            "username": u.get("username") or "",
            "source_market_id": u["market_id"],
        }
        for u in unique_users
    ]
    write_csv_rows(users_raw_path, raw_rows, mode="w")
    logger.info(f"Wrote {len(raw_rows)} raw users to {users_raw_path}")

    # Step 5: Enrich users with positions/trades/activity
    logger.info(f"\n[Step 4/5] Enriching {len(unique_users)} users (this may take a while)...")

    enriched_users = []
    for i, user in enumerate(unique_users, 1):
        if i % 10 == 0 or i == 1:
            logger.info(f"Enriched {i}/{len(unique_users)} users...")

        try:
            enriched = await enrich_user(user)
            enriched_users.append(enriched)
        except Exception as e:
            logger.error(f"Failed to enrich user {user['wallet'][:8]}: {e}")
            continue

        # Rate limiting delay
        await asyncio.sleep(0.2)

    # Write enriched users CSV
    users_enriched_path = output_dir / "users_enriched.csv"
    enriched_rows = [u.to_csv_row() for u in enriched_users]
    write_csv_rows(users_enriched_path, enriched_rows, mode="w")
    logger.info(f"Wrote {len(enriched_rows)} enriched users to {users_enriched_path}")

    # Step 6: Add financial metrics
    logger.info(f"\n[Step 5/5] Adding financial metrics to {len(enriched_rows)} users...")

    # Read enriched CSV
    with users_enriched_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        user_rows = list(reader)

    # Add new field names
    new_fieldnames = list(fieldnames) + [
        "total_cash_volume",
        "realized_pnl",
        "closed_positions_count",
        "winning_positions_count",
        "win_rate"
    ]

    # Enrich with financials
    final_users = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, user_row in enumerate(user_rows, 1):
            if i % 10 == 0 or i == 1:
                logger.info(f"Adding financials {i}/{len(user_rows)}...")

            try:
                enriched_financial = await enrich_user_financials(user_row, client)
                final_users.append(enriched_financial)
            except Exception as e:
                logger.error(f"Failed to add financials for {user_row['wallet'][:8]}: {e}")
                # Add empty financial fields
                user_row.update({
                    "total_cash_volume": 0.0,
                    "realized_pnl": 0.0,
                    "closed_positions_count": 0,
                    "winning_positions_count": 0,
                    "win_rate": 0.0
                })
                final_users.append(user_row)

            # Rate limiting delay
            if i < len(user_rows):
                await asyncio.sleep(0.3)

    # Write final CSV with all metrics
    users_final_path = output_dir / "users_enriched_with_financials.csv"
    with users_final_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(final_users)

    logger.info(f"Wrote {len(final_users)} users with financials to {users_final_path}")

    # Print summary statistics
    logger.info(f"\n=== Summary Statistics ===")
    logger.info(f"Total unique users: {len(final_users)}")
    logger.info(f"Total markets processed: {len(market_ids)}")

    total_pnl = sum(float(u.get("realized_pnl", 0)) for u in final_users)
    users_with_closed = [u for u in final_users if int(u.get("closed_positions_count", 0)) > 0]
    avg_win_rate = (
        sum(float(u.get("win_rate", 0)) for u in users_with_closed) / len(users_with_closed)
        if users_with_closed else 0
    )
    total_cash_vol = sum(float(u.get("total_cash_volume", 0)) for u in final_users)

    logger.info(f"Total Realized PnL: ${total_pnl:,.2f}")
    logger.info(f"Average Win Rate: {avg_win_rate:.2%} (from {len(users_with_closed)} users with closed positions)")
    logger.info(f"Total Cash Volume: ${total_cash_vol:,.2f}")

    logger.info(f"\n=== Output Files ===")
    logger.info(f"Raw users: {users_raw_path}")
    logger.info(f"Enriched users: {users_enriched_path}")
    logger.info(f"Users with financials: {users_final_path}")

    logger.info("\n=== Complete! ===")


if __name__ == "__main__":
    asyncio.run(main())
