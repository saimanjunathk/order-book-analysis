# ORDER BOOK RECONSTRUCTOR
# Takes raw order events and reconstructs the full limit order book
# at every point in time
#
# The limit order book has two sides:
# - Ask side (sell orders): sorted from lowest to highest price
# - Bid side (buy orders):  sorted from highest to lowest price
#
# The "spread" = best ask - best bid
# Market makers profit from this spread
#
# Example order book:
# Ask: 175.03 (100 shares)
#      175.02 (200 shares)
#      175.01 (50 shares)   ← best ask
# ─────────────────────────
#      174.99 (150 shares)  ← best bid
#      174.98 (300 shares)
#      174.97 (200 shares)
# Bid:

import pandas as pd
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrderBookReconstructor:

    def __init__(self, levels: int = 5):
        # levels → how many price levels to track on each side
        # Level 1 = best bid/ask (most important)
        # Level 5 = 5th best bid/ask (shows market depth)
        self.levels = levels

        # Order books stored as dicts: price → size
        self.bids = defaultdict(int)  # buy orders
        self.asks = defaultdict(int)  # sell orders

        self.snapshots = []  # historical snapshots


    # This METHOD processes a single order event
    def process_event(self, event: dict):

        event_type = event["event_type"]
        price      = event["price"]
        size       = event["size"]
        direction  = event["direction"]

        # direction: 1 = ask side (sell), -1 = bid side (buy)
        book = self.asks if direction == 1 else self.bids

        if event_type == 1:
            # New limit order: add to book
            book[price] += size

        elif event_type in [2, 3]:
            # Cancellation: remove from book
            if price in book:
                book[price] = max(0, book[price] - size)
                if book[price] == 0:
                    del book[price]

        elif event_type in [4, 5]:
            # Execution: remove from book (trade happened)
            if price in book:
                book[price] = max(0, book[price] - size)
                if book[price] == 0:
                    del book[price]

        # Take snapshot of current book state
        self._take_snapshot(event["timestamp"], event["mid_price"])


    # This METHOD takes a snapshot of the current order book state
    def _take_snapshot(self, timestamp, mid_price: float):

        snapshot = {"timestamp": timestamp, "mid_price": mid_price}

        # Get top N bid levels (highest prices first)
        bid_prices = sorted(self.bids.keys(), reverse=True)[:self.levels]
        ask_prices = sorted(self.asks.keys())[:self.levels]

        for i in range(self.levels):
            # Bid levels
            if i < len(bid_prices):
                snapshot[f"bid_price_{i+1}"] = bid_prices[i]
                snapshot[f"bid_size_{i+1}"]  = self.bids[bid_prices[i]]
            else:
                snapshot[f"bid_price_{i+1}"] = np.nan
                snapshot[f"bid_size_{i+1}"]  = 0

            # Ask levels
            if i < len(ask_prices):
                snapshot[f"ask_price_{i+1}"] = ask_prices[i]
                snapshot[f"ask_size_{i+1}"]  = self.asks[ask_prices[i]]
            else:
                snapshot[f"ask_price_{i+1}"] = np.nan
                snapshot[f"ask_size_{i+1}"]  = 0

        # Calculate spread
        if bid_prices and ask_prices:
            snapshot["best_bid"]   = bid_prices[0]
            snapshot["best_ask"]   = ask_prices[0]
            snapshot["spread"]     = round(ask_prices[0] - bid_prices[0], 4)
            snapshot["mid_price2"] = round((ask_prices[0] + bid_prices[0]) / 2, 4)
        else:
            snapshot["best_bid"]   = np.nan
            snapshot["best_ask"]   = np.nan
            snapshot["spread"]     = np.nan
            snapshot["mid_price2"] = mid_price

        self.snapshots.append(snapshot)


    # This METHOD reconstructs the full order book from events
    def reconstruct(self, events_df: pd.DataFrame) -> pd.DataFrame:

        logger.info(f"Reconstructing order book from {len(events_df)} events...")

        for _, event in events_df.iterrows():
            self.process_event(event.to_dict())

        snapshots_df = pd.DataFrame(self.snapshots)
        logger.info(f"Generated {len(snapshots_df)} order book snapshots")
        return snapshots_df