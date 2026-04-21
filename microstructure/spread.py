# BID-ASK SPREAD ANALYSIS
# The bid-ask spread is the most fundamental market microstructure measure
# It represents the cost of immediate trading
#
# Types of spread:
# - Quoted spread:    best ask - best bid (what you see)
# - Effective spread: 2 * |trade price - mid price| (what you pay)
# - Realized spread:  measures market maker profitability
#
# Spread components:
# - Inventory cost:   market makers need compensation for holding risk
# - Adverse selection: cost of trading with informed traders
# - Order processing: fixed costs of running a market

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SpreadAnalyzer:

    def __init__(self, events_df: pd.DataFrame, snapshots_df: pd.DataFrame):
        self.events    = events_df
        self.snapshots = snapshots_df


    # This METHOD calculates quoted spread statistics
    def quoted_spread(self) -> pd.DataFrame:

        df = self.snapshots.copy()
        df = df.dropna(subset=["best_bid", "best_ask"])

        # Quoted spread in dollars
        df["quoted_spread"]     = df["best_ask"] - df["best_bid"]

        # Quoted spread in basis points (relative to mid price)
        # 1 basis point = 0.01%
        df["quoted_spread_bps"] = (
            df["quoted_spread"] / df["mid_price"] * 10000
        ).round(4)

        logger.info(f"Avg quoted spread: ${df['quoted_spread'].mean():.4f} "
                   f"({df['quoted_spread_bps'].mean():.2f} bps)")
        return df


    # This METHOD calculates effective spread from trades
    def effective_spread(self) -> pd.DataFrame:

        # Filter for execution events only
        trades = self.events[self.events["event_type"].isin([4, 5])].copy()

        if trades.empty:
            return pd.DataFrame()

        # Effective spread = 2 * |trade price - mid price|
        trades["effective_spread"] = (
            2 * abs(trades["price"] - trades["mid_price"])
        )

        # In basis points
        trades["effective_spread_bps"] = (
            trades["effective_spread"] / trades["mid_price"] * 10000
        ).round(4)

        logger.info(f"Avg effective spread: ${trades['effective_spread'].mean():.4f}")
        return trades


    # This METHOD analyzes spread over time (intraday pattern)
    def intraday_spread(self, freq: str = "5min") -> pd.DataFrame:

        df = self.quoted_spread()
        df = df.set_index("timestamp")

        # Resample to desired frequency
        intraday = df["quoted_spread_bps"].resample(freq).mean().reset_index()
        intraday.columns = ["time", "avg_spread_bps"]

        return intraday.dropna()