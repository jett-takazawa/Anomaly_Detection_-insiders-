"""
Add financial metrics to enriched user data.

Enhances users_enriched.csv with:
- Total cash volume (sum of USD values from all trades)
- Realized PnL (from closed positions)
- Win rate (percentage of profitable closed positions)
- Closed positions count and winning positions count
"""

import asyncio
import csv
import logging
from pathlib import Path

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_API_BASE_URL = "https://data-api.polymarket.com"
REQUEST_DELAY = 0.3  # Delay between API calls to respect rate limits


async def fetch_trades_with_usd(wallet: str, client: httpx.AsyncClient) -> float:
    """
    Fetch all trades for a user and calculate total cash volume.

    Args:
        wallet: User wallet address
        client: HTTP client for making requests

    Returns:
        Total cash volume (sum of amountUSD from all trades)
    """
    url = f"{DATA_API_BASE_URL}/trades"
    params = {
        "user": wallet,
        "limit": 1000,
    }

    try:
        response = await client.get(url, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        # Handle both list and dict responses
        items = data if isinstance(data, list) else data.get("data", [])

        total_cash_volume = 0.0

        for item in items:
            # Calculate USD amount from size * price
            price = float(item.get("price", 0) or 0)
            size = float(item.get("size", 0) or 0)
            amount_usd = price * size

            total_cash_volume += amount_usd

        return total_cash_volume

    except Exception as e:
        logger.debug(f"Failed to fetch trades for {wallet[:8]}: {e}")
        return 0.0


async def fetch_closed_positions(
    wallet: str,
    client: httpx.AsyncClient
) -> tuple[float, int, int]:
    """
    Fetch closed positions for a user to calculate PnL and win rate.

    Args:
        wallet: User wallet address
        client: HTTP client for making requests

    Returns:
        Tuple of (realized_pnl, closed_positions_count, winning_positions_count)
    """
    url = f"{DATA_API_BASE_URL}/closed-positions"
    params = {
        "user": wallet,
        "limit": 1000,
    }

    try:
        response = await client.get(url, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        # Handle both list and dict responses
        items = data if isinstance(data, list) else data.get("data", [])

        realized_pnl = 0.0
        closed_count = len(items)
        winning_count = 0

        for item in items:
            # PnL - use camelCase field name from API
            pnl = float(item.get("realizedPnl", 0) or 0)
            realized_pnl += pnl

            # Determine if position was a winner based on positive PnL
            if pnl > 0:
                winning_count += 1

        return realized_pnl, closed_count, winning_count

    except Exception as e:
        logger.debug(f"Failed to fetch closed positions for {wallet[:8]}: {e}")
        return 0.0, 0, 0


async def enrich_user_financials(
    user_row: dict,
    client: httpx.AsyncClient
) -> dict:
    """
    Enrich a single user with financial metrics.

    Args:
        user_row: Dictionary containing user data from CSV
        client: HTTP client for making requests

    Returns:
        Enhanced user row with financial metrics
    """
    wallet = user_row["wallet"]
    logger.info(f"Enriching financials for user {wallet[:8]}...")

    # Fetch trades and closed positions in parallel
    total_cash_volume, (realized_pnl, closed_count, winning_count) = await asyncio.gather(
        fetch_trades_with_usd(wallet, client),
        fetch_closed_positions(wallet, client)
    )

    # Calculate win rate
    win_rate = (winning_count / closed_count) if closed_count > 0 else 0.0

    # Add new fields to user row
    user_row["total_cash_volume"] = round(total_cash_volume, 6)
    user_row["realized_pnl"] = round(realized_pnl, 6)
    user_row["closed_positions_count"] = closed_count
    user_row["winning_positions_count"] = winning_count
    user_row["win_rate"] = round(win_rate, 4)

    return user_row


async def process_users_csv(
    input_csv: Path,
    output_csv: Path
) -> None:
    """
    Process all users from input CSV and add financial metrics.

    Args:
        input_csv: Path to users_enriched.csv
        output_csv: Path to output CSV with financial metrics
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Read all users
    logger.info(f"Reading users from {input_csv}")
    users = []
    with input_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            users.append(row)

    logger.info(f"Found {len(users)} users to enrich")

    if not users:
        logger.warning("No users found in input CSV")
        return

    # Add new field names
    new_fieldnames = list(fieldnames) + [
        "total_cash_volume",
        "realized_pnl",
        "closed_positions_count",
        "winning_positions_count",
        "win_rate"
    ]

    # Process users
    enriched_users = []
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, user in enumerate(users, 1):
            logger.info(f"Processing user {i}/{len(users)}")

            enriched = await enrich_user_financials(user, client)
            enriched_users.append(enriched)

            # Rate limiting delay
            if i < len(users):
                await asyncio.sleep(REQUEST_DELAY)

    # Write output CSV
    logger.info(f"Writing enriched data to {output_csv}")
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(enriched_users)

    logger.info(f"Successfully enriched {len(enriched_users)} users")

    # Print summary statistics
    total_pnl = sum(float(u["realized_pnl"]) for u in enriched_users)
    avg_win_rate = sum(float(u["win_rate"]) for u in enriched_users) / len(enriched_users)
    total_cash_vol = sum(float(u["total_cash_volume"]) for u in enriched_users)

    logger.info(f"\n=== Summary Statistics ===")
    logger.info(f"Total Realized PnL: ${total_pnl:,.2f}")
    logger.info(f"Average Win Rate: {avg_win_rate:.2%}")
    logger.info(f"Total Cash Volume: ${total_cash_vol:,.2f}")


async def main():
    """Main entry point."""
    # Default paths
    script_dir = Path(__file__).parent
    input_csv = script_dir / "outputs" / "users_enriched.csv"
    output_csv = script_dir / "outputs" / "users_enriched_with_financials.csv"

    logger.info("Starting financial metrics enrichment")
    logger.info(f"Input: {input_csv}")
    logger.info(f"Output: {output_csv}")

    await process_users_csv(input_csv, output_csv)

    logger.info("Financial metrics enrichment complete!")


if __name__ == "__main__":
    asyncio.run(main())
