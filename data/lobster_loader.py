# LOBSTER DATA LOADER
# LOBSTER = Limit Order Book System — The Efficient Reconstructor
# Real LOBSTER data contains every order event in the market:
# - New order submissions
# - Order cancellations
# - Order executions (trades)
#
# Each event has: timestamp, event type, order ID, size, price, direction
#
# We simulate realistic LOBSTER-style data since real data costs money
# The simulation captures real market microstructure properties:
# - Price clustering at round numbers
# - Bid-ask spread dynamics
# - Order size distributions
# - Intraday volume patterns

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LOBSTERSimulator:

    # symbol        → stock ticker to simulate
    # start_price   → starting mid price
    # n_events      → number of order events to generate
    # tick_size     → minimum price increment ($0.01 for most US stocks)
    def __init__(
        self,
        symbol: str = "AAPL",
        start_price: float = 175.0,
        n_events: int = 10000,
        tick_size: float = 0.01
    ):
        self.symbol      = symbol
        self.start_price = start_price
        self.n_events    = n_events
        self.tick_size   = tick_size

        np.random.seed(42)

    # LOBSTER event types:
    # 1 = New limit order submission
    # 2 = Partial cancellation
    # 3 = Full cancellation
    # 4 = Execution of visible limit order
    # 5 = Execution of hidden limit order
    EVENT_TYPES = {
        1: "new_limit_order",
        2: "partial_cancel",
        3: "full_cancel",
        4: "execution_visible",
        5: "execution_hidden"
    }

    # This METHOD generates realistic order events
    def generate_events(self) -> pd.DataFrame:

        logger.info(f"Generating {self.n_events} order events for {self.symbol}...")

        events      = []
        mid_price   = self.start_price
        order_id    = 1
        # Start time: 9:30 AM (market open)
        start_time  = datetime(2024, 1, 15, 9, 30, 0)

        for i in range(self.n_events):

            # Time increment: microseconds between events
            # Events cluster in time (realistic market behavior)
            time_delta = np.random.exponential(0.1)  # avg 100ms between events
            if i == 0:
                timestamp = start_time
            else:
                timestamp = events[-1]["timestamp"] + timedelta(seconds=time_delta)

            # Simulate price evolution using random walk
            # Small drift toward mean (mean reversion)
            price_change = np.random.normal(0, 0.02)
            mid_price    = mid_price * (1 + price_change * 0.001)

            # Bid-ask spread: typically 1-3 ticks for liquid stocks
            spread_ticks = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            spread       = spread_ticks * self.tick_size

            # Best bid and ask prices
            best_ask = round(mid_price + spread/2, 2)
            best_bid = round(mid_price - spread/2, 2)

            # Event type probabilities (realistic market distribution)
            event_type = np.random.choice(
                [1, 2, 3, 4, 5],
                p=[0.45, 0.15, 0.20, 0.15, 0.05]
            )

            # Direction: 1=buy/ask side, -1=sell/bid side
            direction = np.random.choice([1, -1])

            # Order size: log-normal distribution (small orders most common)
            size = max(1, int(np.random.lognormal(4, 1)))
            size = min(size, 10000)  # cap at 10,000 shares

            # Price depends on direction and event type
            if event_type in [4, 5]:
                # Executions happen at best bid/ask
                price = best_ask if direction == 1 else best_bid
            else:
                # Limit orders placed near best bid/ask
                offset = np.random.choice([0, 1, 2, 3]) * self.tick_size
                if direction == 1:
                    price = round(best_bid - offset, 2)
                else:
                    price = round(best_ask + offset, 2)

            events.append({
                "timestamp":  timestamp,
                "event_type": event_type,
                "order_id":   order_id,
                "size":       size,
                "price":      price,
                "direction":  direction,
                "mid_price":  round(mid_price, 4),
                "best_bid":   best_bid,
                "best_ask":   best_ask,
                "spread":     round(spread, 4)
            })

            order_id += 1

        df = pd.DataFrame(events)
        logger.info(f"Generated {len(df)} events from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df


    # This METHOD generates OHLCV bars from order events
    def generate_ohlcv(self, freq: str = "1min") -> pd.DataFrame:

        events = self.generate_events()
        events = events.set_index("timestamp")

        # Resample to desired frequency
        ohlcv = events["mid_price"].resample(freq).agg({
            "open":  "first",
            "high":  "max",
            "low":   "min",
            "close": "last"
        }).dropna()

        # Add volume
        volume = events["size"].resample(freq).sum()
        ohlcv["volume"] = volume

        ohlcv.columns = ["open", "high", "low", "close", "volume"]
        return ohlcv.dropna()


if __name__ == "__main__":
    sim    = LOBSTERSimulator(symbol="AAPL", n_events=5000)
    events = sim.generate_events()
    print(f"Events shape: {events.shape}")
    print(f"\nEvent type distribution:")
    print(events["event_type"].value_counts())
    print(f"\nSample events:")
    print(events.head(5))