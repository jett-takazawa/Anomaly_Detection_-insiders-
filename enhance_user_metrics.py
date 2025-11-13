"""
Enhance user data with dollar-denominated metrics, PnL, and win rate.

Reads existing users_enriched_earnings_only.csv and adds:
- Total dollar volume (not just share volume)
- Average trade size in dollars
- Profit/Loss (realized and unrealized)
- Win rate (% of profitable trades)
- Position values in dollars
"""

import asyncio
import csv
import logging
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_API_BASE_URL = "https://data-api.polymarket.com"


async def fetch_user_trades_with_prices(wallet: str) -> tuple[dict, dict]:
    """
    Fetch user trades with price data to calculate dollar metrics.

    Returns tuple of (metrics_dict, open_positions_dict) where:
    - metrics_dict contains: total_dollar_volume, avg_trade_dollars, realized_pnl, win_rate, etc.
    - open_positions_dict tracks remaining open positions with cost basis for unrealized PnL
    """
    url = f"{DATA_API_BASE_URL}/trades"
    params = {"user": wallet, "limit": 1000}

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        items = data if isinstance(data, list) else data.get("data", [])

        if not items:
            return {
                "total_dollar_volume": 0.0,
                "avg_trade_dollars": 0.0,
                "realized_pnl": 0.0,
                "win_rate": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_trades": 0,
            }

        total_dollar_volume = 0.0
        trade_count = 0

        # Track positions to calculate PnL
        # Key: (market_id, outcome), Value: {"shares": X, "cost_basis": Y}
        positions = {}
        realized_pnl = 0.0
        winning_trades = 0
        losing_trades = 0

        for item in items:
            # Get trade details
            size = float(item.get("size", 0) or 0)
            price = float(item.get("price", 0) or 0)
            side = item.get("side", "").upper()
            market_id = item.get("market") or item.get("marketId", "")
            outcome = item.get("outcome", "")

            # Calculate dollar volume for this trade
            dollar_amount = size * price
            total_dollar_volume += dollar_amount
            trade_count += 1

            # Track position for PnL calculation
            position_key = (market_id, outcome)

            if side == "BUY":
                # Add to position
                if position_key not in positions:
                    positions[position_key] = {"shares": 0.0, "cost_basis": 0.0}

                positions[position_key]["shares"] += size
                positions[position_key]["cost_basis"] += dollar_amount

            elif side == "SELL":
                # Reduce position and realize PnL
                if position_key in positions and positions[position_key]["shares"] > 0:
                    # Calculate PnL for this sell
                    avg_cost = positions[position_key]["cost_basis"] / positions[position_key]["shares"]
                    sell_proceeds = dollar_amount
                    cost_of_shares_sold = avg_cost * size
                    trade_pnl = sell_proceeds - cost_of_shares_sold

                    realized_pnl += trade_pnl

                    # Track winning vs losing trades
                    if trade_pnl > 0:
                        winning_trades += 1
                    elif trade_pnl < 0:
                        losing_trades += 1

                    # Update position
                    positions[position_key]["shares"] -= size
                    if positions[position_key]["shares"] > 0:
                        positions[position_key]["cost_basis"] = (
                            avg_cost * positions[position_key]["shares"]
                        )
                    else:
                        # Position fully closed
                        del positions[position_key]

        avg_trade_dollars = total_dollar_volume / trade_count if trade_count > 0 else 0.0

        # Win rate calculation
        total_closed_trades = winning_trades + losing_trades
        win_rate = (winning_trades / total_closed_trades * 100) if total_closed_trades > 0 else 0.0

        metrics = {
            "total_dollar_volume": round(total_dollar_volume, 2),
            "avg_trade_dollars": round(avg_trade_dollars, 2),
            "realized_pnl": round(realized_pnl, 2),
            "win_rate": round(win_rate, 2),
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "total_trades": trade_count,
        }

        return metrics, positions

    except Exception as e:
        logger.warning(f"Failed to fetch trades for {wallet[:8]}: {e}")
        return {
            "total_dollar_volume": 0.0,
            "avg_trade_dollars": 0.0,
            "realized_pnl": 0.0,
            "win_rate": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_trades": 0,
        }, {}


async def fetch_user_positions_dollars(wallet: str, open_positions: dict) -> dict:
    """
    Fetch user positions with dollar values using open positions from trade history.

    Args:
        wallet: User wallet
        open_positions: Dict of open positions from trade history with cost basis

    Returns dict with:
    - position_value_dollars (current market value)
    - unrealized_pnl
    """
    url = f"{DATA_API_BASE_URL}/positions"
    params = {
        "user": wallet,
        "limit": 1000,
        "sizeThreshold": 1,
        "sortBy": "TOKENS",
        "sortDirection": "DESC",
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        items = data if isinstance(data, list) else data.get("data", [])

        position_value_dollars = 0.0
        unrealized_pnl = 0.0

        for item in items:
            # Position size (shares)
            size = float(item.get("size", 0) or item.get("tokens", 0) or 0)

            # Current market price
            current_price = float(item.get("price", 0) or 0)

            # Calculate current value
            current_value = size * current_price
            position_value_dollars += current_value

            # Get cost basis from our tracked positions
            market_id = item.get("market") or item.get("marketId", "")
            outcome = item.get("outcome", "")
            position_key = (market_id, outcome)

            if position_key in open_positions and open_positions[position_key]["shares"] > 0:
                avg_cost = open_positions[position_key]["cost_basis"] / open_positions[position_key]["shares"]
                initial_cost = size * avg_cost
                unrealized_pnl += (current_value - initial_cost)

        return {
            "position_value_dollars": round(position_value_dollars, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
        }

    except Exception as e:
        logger.warning(f"Failed to fetch positions for {wallet[:8]}: {e}")
        return {
            "position_value_dollars": 0.0,
            "unrealized_pnl": 0.0,
        }


async def enhance_user(row: dict) -> dict:
    """Enhance a single user row with dollar metrics."""
    wallet = row["wallet"]

    logger.info(f"Enhancing {wallet[:8]}...")

    # Fetch dollar-denominated trade metrics and open positions
    trade_metrics, open_positions = await fetch_user_trades_with_prices(wallet)

    # Fetch dollar-denominated position metrics using tracked cost basis
    position_metrics = await fetch_user_positions_dollars(wallet, open_positions)

    # Calculate total PnL (realized + unrealized)
    total_pnl = trade_metrics["realized_pnl"] + position_metrics["unrealized_pnl"]

    # Add new fields to row
    enhanced_row = row.copy()
    enhanced_row.update({
        "total_dollar_volume": trade_metrics["total_dollar_volume"],
        "avg_trade_dollars": trade_metrics["avg_trade_dollars"],
        "realized_pnl": trade_metrics["realized_pnl"],
        "unrealized_pnl": position_metrics["unrealized_pnl"],
        "total_pnl": round(total_pnl, 2),
        "win_rate_percent": trade_metrics["win_rate"],
        "winning_trades": trade_metrics["winning_trades"],
        "losing_trades": trade_metrics["losing_trades"],
        "position_value_dollars": position_metrics["position_value_dollars"],
    })

    return enhanced_row


async def main():
    input_path = Path("outputs/users_enriched_earnings_only.csv")
    output_path = Path("outputs/users_enriched_earnings_only_enhanced.csv")

    logger.info(f"Reading users from {input_path}...")

    with input_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        users = list(reader)
        headers = reader.fieldnames

    logger.info(f"Found {len(users)} users to enhance")

    # Add new headers
    new_headers = list(headers) + [
        "total_dollar_volume",
        "avg_trade_dollars",
        "realized_pnl",
        "unrealized_pnl",
        "total_pnl",
        "win_rate_percent",
        "winning_trades",
        "losing_trades",
        "position_value_dollars",
    ]

    # Enhance each user
    enhanced_users = []
    for i, user in enumerate(users, 1):
        logger.info(f"Processing {i}/{len(users)}: {user['wallet'][:10]}...")

        enhanced = await enhance_user(user)
        enhanced_users.append(enhanced)

        # Rate limit: 1 user per second
        await asyncio.sleep(1.0)

    # Write enhanced CSV
    logger.info(f"Writing enhanced data to {output_path}...")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_headers)
        writer.writeheader()
        writer.writerows(enhanced_users)

    logger.info(f"Done! Enhanced {len(enhanced_users)} users")
    logger.info(f"Output: {output_path}")

    # Show summary stats
    total_pnls = [float(u.get("total_pnl", 0)) for u in enhanced_users]
    win_rates = [float(u.get("win_rate_percent", 0)) for u in enhanced_users if float(u.get("win_rate_percent", 0)) > 0]

    if total_pnls:
        print(f"\n=== Summary Statistics ===")
        print(f"Average PnL: ${sum(total_pnls) / len(total_pnls):.2f}")
        print(f"Max PnL: ${max(total_pnls):.2f}")
        print(f"Min PnL: ${min(total_pnls):.2f}")

    if win_rates:
        print(f"Average Win Rate: {sum(win_rates) / len(win_rates):.2f}%")
        print(f"Users with >60% win rate: {len([r for r in win_rates if r > 60])}")


if __name__ == "__main__":
    asyncio.run(main())
