# Anomaly Detection – Potential “Insiders” on Polymarket

This project explores whether we can use **anomaly detection** to flag wallets on Polymarket that *behave like* potential “insiders,” especially around earnings markets. It is **not** about proving illegal insider trading, but about surfacing **unusual, high‑signal behavior** that might warrant deeper investigation.

---

## Core Idea

Most wallets on Polymarket behave in a “normal” way:
- They trade many different markets.
- They have noisy PnL.
- Their wins and losses look like regular speculation.

A small subset of wallets, however, may:
- Trade **very selectively** (only a few markets).
- Focus on **specific earnings markets**.
      - Normal wallets will trade on **multiple** earnings markets
- Show **unusually strong, consistent PnL** in those markets.
- Enter and exit positions at **suspiciously good timing**.

This repo treats those patterns as **statistical anomalies**. Instead of starting with labeled “insiders,” it uses unsupervised / semi‑supervised methods to **rank wallets by how abnormal their behavior is** compared with the broader population.

---

## What the Project Actually Does

At a high level, the pipeline is:

1. **Ingest Polymarket-style wallet & trade data**
   - Wallet IDs, usernames, timestamps.
   - Market titles (with a focus on earnings markets).
   - Trade history (buys, sells, taker/maker).
   - Position sizes, volume, and rough PnL.

2. **Engineer wallet-level features**
   Examples of features this project explores:
   - Number of markets traded vs. **earnings-only focus**.
   - Concentration of risk in a few markets or tickers.
   - Asymmetry between buy/sell behavior.
   - Total and average trade size.
   - First/last activity timestamps and active window.
   - “Missingness” patterns (e.g., someone only appears for one key event and vanishes).

3. **Apply anomaly detection models**
   The two main modeling notebooks live at the root of the repo:
   - `DB_scan_rules_regex.ipynb` – DBSCAN-based clustering + rule/regex filters on titles and patterns.
   - `IsolationForest_rules_regex.ipynb` – Isolation Forest anomaly scores with rule-based post-filtering.

   These models don’t claim “this wallet is an insider.” Instead, they answer:
   > “Compared to typical behavior in the dataset, **how strange is this wallet?**”

4. **Rank and save the suspicious cases**
   Final processed outputs and ranked wallets are stored in:

   ```text
   test5_AWESOME(no_crypto)/
   ```

   This folder contains the **refined datasets and model results** for analysis, including:
   - Cleaned wallet-level tables.
   - Anomaly scores from Isolation Forest / DBSCAN.
   - Filtered subsets of “most unusual” wallets.

---

## How to Read the Results

If you open the two primary notebooks:

- **`DB_scan_rules_regex.ipynb`**
  - Experiments with clustering and grouping wallets based on feature similarity.
  - Uses regex and rule-based filters on market titles (e.g., detecting earnings markets, ticker patterns).
  - Helps identify tight clusters of “normal” wallets and outliers that sit alone or in tiny clusters.

- **`IsolationForest_rules_regex.ipynb`**
  - Trains an Isolation Forest on the engineered features to produce an anomaly score per wallet.
  - Higher scores ≈ “this wallet’s behavior is rare given the data distribution.”
  - Rules and regex layers are then applied on top to emphasize behaviors that *look* insider-like (e.g., multiple successful, concentrated earnings bets).

The **`test5_AWESOME(no_crypto)`** directory stores the csv's that these notebooks rely on or produce. Think of it as the “anomalies” where you can sort, filter, and manually inspect the top-ranked wallets.

---

## What This Project *Is* and *Is Not*

**This project *is*:**  
- A technical exploration of using anomaly detection on prediction market data.  
- A way to generate **ranked leads** for deeper human analysis.  
- A framework for thinking about “insider-like” behavior using features, not labels.

**This project is *not*:**  
- A legal or regulatory determination of insider trading.  
- A guarantee that any flagged wallet is doing anything illegal or unethical.  
- A production-ready compliance system (yet).

The whole point is to **push the frontier** of what unsupervised ML can reveal in small, noisy markets where labels are rare or nonexistent.

---

## Next Directions

Some natural extensions of this work include:

- Incorporating **LLM-based text scoring** of market titles to better detect earnings-related and thematically-linked markets per wallet.
- Adding **time-series features** (entry/exit timing vs. information release).
- Comparing anomaly scores against **ground truth events** once any are available.
- Deploying a lightweight pipeline that periodically re-scores wallets as new trades come in.

For now, this repository serves as a **sandbox for ideas** around anomaly-based “insider hunter” tooling on Polymarket-style data.
