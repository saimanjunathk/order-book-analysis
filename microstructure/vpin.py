# VPIN = Volume-Synchronized Probability of Informed Trading
# Developed by Easley, Lopez de Prado, and O'Hara (2012)
#
# VPIN measures "order flow toxicity" — how likely you are to
# be trading against informed traders who know something you don't
#
# High VPIN → high probability of informed trading → dangerous for market makers
# Low VPIN  → mostly uninformed noise trading → safe for market makers
#
# VPIN was used to predict the 2010 Flash Crash!
# It spiked to 0.92 (very high) 1 hour before the crash
#
# Formula:
# VPIN = |buy_volume - sell_volume| / total_volume
# Calculated over fixed volume buckets (not time buckets)

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VPINCalculator:

    # n_buckets    → number of volume buckets to use
    # window       → rolling window for VPIN calculation
    def __init__(self, n_buckets: int = 50, window: int = 50):
        self.n_buckets = n_buckets
        self.window    = window


    # This METHOD classifies trades as buys or sells
    # Uses tick rule: if price went up → buy, if down → sell
    def classify_trades(self, events_df: pd.DataFrame) -> pd.DataFrame:

        trades = events_df[events_df["event_type"].isin([4, 5])].copy()

        if trades.empty:
            return pd.DataFrame()

        # Tick rule: compare trade price to previous trade price
        trades["prev_price"] = trades["price"].shift(1)
        trades["price_change"] = trades["price"] - trades["prev_price"]

        # Classify: positive change = buy, negative = sell, zero = carry forward
        trades["trade_direction"] = np.where(
            trades["price_change"] > 0,  1,   # buy
            np.where(
                trades["price_change"] < 0, -1,  # sell
                0  # unchanged (will forward fill)
            )
        )

        # Forward fill unchanged classifications
        trades["trade_direction"] = trades["trade_direction"].replace(0, np.nan).ffill().fillna(1)

        trades["buy_volume"]  = trades["size"] * (trades["trade_direction"] ==  1).astype(int)
        trades["sell_volume"] = trades["size"] * (trades["trade_direction"] == -1).astype(int)

        return trades


    # This METHOD calculates VPIN
    def calculate(self, events_df: pd.DataFrame) -> pd.DataFrame:

        trades = self.classify_trades(events_df)

        if trades.empty:
            return pd.DataFrame()

        # Calculate total volume to determine bucket size
        total_volume = trades["size"].sum()
        bucket_size  = total_volume / self.n_buckets

        logger.info(f"Total volume: {total_volume:,} | Bucket size: {bucket_size:.0f}")

        # Fill volume buckets
        buckets      = []
        current_buy  = 0
        current_sell = 0
        current_vol  = 0

        for _, trade in trades.iterrows():
            current_buy  += trade["buy_volume"]
            current_sell += trade["sell_volume"]
            current_vol  += trade["size"]

            if current_vol >= bucket_size:
                buckets.append({
                    "timestamp":   trade["timestamp"],
                    "buy_volume":  current_buy,
                    "sell_volume": current_sell,
                    "total_volume": current_vol,
                    "imbalance":   abs(current_buy - current_sell)
                })
                current_buy  = 0
                current_sell = 0
                current_vol  = 0

        if not buckets:
            return pd.DataFrame()

        buckets_df = pd.DataFrame(buckets)

        # VPIN = rolling average of |buy - sell| / total_volume
        buckets_df["vpin"] = (
            buckets_df["imbalance"].rolling(self.window).mean() /
            bucket_size
        ).round(4)

        # VPIN interpretation
        buckets_df["toxicity"] = pd.cut(
            buckets_df["vpin"].fillna(0),
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["Low", "Medium", "High", "Very High"]
        )

        logger.info(f"Calculated VPIN for {len(buckets_df)} buckets")
        return buckets_df