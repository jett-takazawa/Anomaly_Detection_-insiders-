"""
Filter users_raw.csv to only include users from earnings markets.

Reads rated_markets.csv to get valid earnings market IDs, then filters
users_raw.csv to only keep users who participated in those markets.
"""

import csv
import re
from pathlib import Path


def is_earnings_market(title: str) -> bool:
    """Check if market title matches earnings pattern."""
    pattern = r"will\s+.+?\s+\([a-z]{1,5}\)\s+beat\s+quarterly\s+earnings"
    return bool(re.search(pattern, title.lower()))


def main():
    # Paths
    markets_path = Path("outputs/rated_markets.csv")
    users_input_path = Path("outputs2/users_raw.csv")
    users_output_path = Path("outputs2/users_raw_filtered.csv")

    print(f"Reading markets from {markets_path}...")

    # Get valid earnings market IDs
    valid_market_ids = set()
    with markets_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = row.get("title", "")
            market_id = row.get("market_id", "")

            if is_earnings_market(title):
                valid_market_ids.add(market_id)
                print(f"  + {title[:60]}...")

    print(f"\nFound {len(valid_market_ids)} valid earnings markets")

    # Filter users
    print(f"\nReading users from {users_input_path}...")

    filtered_users = []
    total_users = 0

    with users_input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        for row in reader:
            total_users += 1
            source_market_id = row.get("source_market_id", "")

            if source_market_id in valid_market_ids:
                filtered_users.append(row)

    print(f"Total users: {total_users}")
    print(f"Filtered users (from earnings markets): {len(filtered_users)}")
    print(f"Removed: {total_users - len(filtered_users)}")

    # Write filtered results
    print(f"\nWriting filtered users to {users_output_path}...")

    with users_output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(filtered_users)

    print(f"Done! Filtered users saved to {users_output_path}")


if __name__ == "__main__":
    main()
