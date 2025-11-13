"""
Fetch enriched data for specific wallet addresses.

Usage:
    uv run python fetch_specific_users.py <wallet1> <wallet2> ...
"""

import asyncio
import csv
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_fetch.fetch_user_data import enrich_user


async def fetch_specific_users(wallets: list[str], output_file: str = "specific_users_enriched.csv"):
    """Fetch and enrich specific users by wallet address."""

    print(f"Fetching data for {len(wallets)} users...")

    enriched_users = []

    for i, wallet in enumerate(wallets, 1):
        print(f"\n[{i}/{len(wallets)}] Enriching wallet: {wallet}")

        # Create a minimal user identity dict
        user_identity = {
            "wallet": wallet.lower(),
            "username": None,  # Will be discovered from API
            "market_id": "manual_fetch",  # Marker for manual fetch
        }

        try:
            enriched_user = await enrich_user(user_identity)
            enriched_users.append(enriched_user)
            print(f"  [OK] Completed: {enriched_user.username or wallet[:16]}...")
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

    # Write to CSV
    output_path = Path(__file__).parent / "outputs" / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if enriched_users:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=enriched_users[0].to_csv_row().keys())
            writer.writeheader()
            for user in enriched_users:
                writer.writerow(user.to_csv_row())

        print(f"\n[OK] Wrote {len(enriched_users)} enriched users to: {output_path}")

        # Print summary
        print("\n=== Summary ===")
        for user in enriched_users:
            print(f"\nWallet: {user.wallet[:16]}...")
            print(f"Username: {user.username or 'N/A'}")
            print(f"Positions: {user.positions_count} across {user.active_markets_count} markets")
            print(f"Trades: {user.trades_count} ({user.buy_trades_count} buy, {user.sell_trades_count} sell)")
            print(f"Volume: ${user.total_volume:.2f}")
            print(f"Avg Trade Size: ${user.avg_trade_size:.2f}")
            if user.first_trade_ts:
                print(f"First Trade: {user.first_trade_ts.strftime('%Y-%m-%d')}")
            if user.last_trade_ts:
                print(f"Last Trade: {user.last_trade_ts.strftime('%Y-%m-%d')}")
    else:
        print("\n[ERROR] No users enriched successfully")

    return enriched_users


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python fetch_specific_users.py <wallet1> <wallet2> ...")
        sys.exit(1)

    wallets = sys.argv[1:]
    asyncio.run(fetch_specific_users(wallets))
