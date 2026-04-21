# SHORT-HORIZON PRICE PREDICTOR
# Uses microstructure features to predict price direction
# over the next 1-5 seconds
#
# Features used:
# - Order flow imbalance (OFI)
# - Book imbalance
# - Bid-ask spread
# - Recent price momentum
# - Volume

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class PricePredictor:

    def __init__(self):
        self.model   = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler      = StandardScaler()
        self.is_trained  = False
        self.feature_names = None


    # This METHOD builds features from order book snapshots
    def build_features(self, snapshots_df: pd.DataFrame) -> pd.DataFrame:

        df = snapshots_df.copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Use mid_price for calculations
        price_col = "mid_price" if "mid_price" in df.columns else "mid_price2"

        # ── Price features ──
        # Short-term momentum (last 5 ticks)
        df["momentum_5"]  = df[price_col].pct_change(5)
        df["momentum_10"] = df[price_col].pct_change(10)

        # Price volatility (rolling std)
        df["volatility"]  = df[price_col].rolling(10).std()

        # ── Order book features ──
        if "spread" in df.columns:
            df["spread_normalized"] = (
                df["spread"] / df[price_col]
            ).round(6)

        # Book imbalance
        if "bid_size_1" in df.columns and "ask_size_1" in df.columns:
            bid_sz = df["bid_size_1"].fillna(0)
            ask_sz = df["ask_size_1"].fillna(0)
            df["book_imbalance"] = (
                (bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-8)
            ).round(4)

            # Total depth at level 1
            df["depth_level1"] = bid_sz + ask_sz

        # ── Target variable ──
        # 1 = price goes up in next 5 ticks
        # 0 = price goes down or stays same
        future_return = df[price_col].shift(-5) - df[price_col]
        df["target"] = (future_return > 0).astype(int)

        return df


    # This METHOD trains the prediction model
    def train(self, snapshots_df: pd.DataFrame) -> dict:

        df = self.build_features(snapshots_df)

        feature_cols = [
            c for c in [
                "momentum_5", "momentum_10", "volatility",
                "spread_normalized", "book_imbalance", "depth_level1"
            ] if c in df.columns
        ]

        self.feature_names = feature_cols

        # Remove rows with NaN
        df = df[feature_cols + ["target"]].dropna()

        if len(df) < 100:
            logger.warning("Not enough data to train")
            return {}

        X = df[feature_cols]
        y = df["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Price predictor accuracy: {accuracy:.4f}")

        # Feature importance
        importance_df = pd.DataFrame({
            "feature":    feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)

        return {
            "accuracy":   accuracy,
            "importance": importance_df,
            "n_samples":  len(df),
            "report":     classification_report(y_test, y_pred)
        }