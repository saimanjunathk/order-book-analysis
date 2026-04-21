# ORDER FLOW IMBALANCE (OFI)
# OFI measures the imbalance between buying and selling pressure
# It's one of the strongest short-term price predictors known
#
# Key insight: if more buyers than sellers → price goes up
#
# OFI formula:
# OFI = (bid_size_change_when_bid_increases) -
#       (bid_size_change_when_bid_decreases) -
#       (ask_size_change_when_ask_decreases) +
#       (ask_size_change_when_ask_increases)
#
# Simplified version we use:
# OFI = buy_volume - sell_volume (per time period)
#
# Research shows OFI explains ~80% of price changes at high frequency!

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class OrderFlowImbalance:

    def __init__(self, events_df: pd.DataFrame):
        self.events = events_df


    # This METHOD calculates OFI per time period
    def calculate_ofi(self, freq: str = "1min") -> pd.DataFrame:

        df = self.events.copy()
        df = df.set_index("timestamp")

        # Buy volume = executions on ask side (direction=1)
        # Sell volume = executions on bid side (direction=-1)
        executions = df[df["event_type"].isin([4, 5])].copy()

        if executions.empty:
            return pd.DataFrame()

        # Buy and sell volumes
        buy_vol  = executions[executions["direction"] ==  1]["size"].resample(freq).sum()
        sell_vol = executions[executions["direction"] == -1]["size"].resample(freq).sum()

        ofi_df = pd.DataFrame({
            "buy_volume":  buy_vol,
            "sell_volume": sell_vol
        }).fillna(0)

        # OFI = buy - sell
        ofi_df["ofi"] = ofi_df["buy_volume"] - ofi_df["sell_volume"]

        # Normalized OFI (-1 to 1)
        total_vol = ofi_df["buy_volume"] + ofi_df["sell_volume"]
        ofi_df["ofi_normalized"] = (
            ofi_df["ofi"] / (total_vol + 1e-8)
        ).round(4)

        # OFI signal: positive = buying pressure, negative = selling pressure
        ofi_df["signal"] = np.where(
            ofi_df["ofi_normalized"] > 0.1,  "buy_pressure",
            np.where(
                ofi_df["ofi_normalized"] < -0.1, "sell_pressure",
                "neutral"
            )
        )

        logger.info(f"Calculated OFI for {len(ofi_df)} periods")
        return ofi_df.reset_index()


    # This METHOD calculates order book imbalance (different from OFI)
    # Uses the sizes at best bid/ask to measure instantaneous imbalance
    def book_imbalance(self, snapshots_df: pd.DataFrame) -> pd.DataFrame:

        df = snapshots_df.copy()

        # Book imbalance = (bid_size - ask_size) / (bid_size + ask_size)
        # Range: -1 (all asks) to +1 (all bids)
        bid_size = df["bid_size_1"].fillna(0)
        ask_size = df["ask_size_1"].fillna(0)

        df["book_imbalance"] = (
            (bid_size - ask_size) /
            (bid_size + ask_size + 1e-8)
        ).round(4)

        return df