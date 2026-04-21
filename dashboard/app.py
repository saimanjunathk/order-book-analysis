import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lobster_loader import LOBSTERSimulator
from orderbook.reconstructor import OrderBookReconstructor
from microstructure.spread import SpreadAnalyzer
from microstructure.order_flow_imbalance import OrderFlowImbalance
from microstructure.vpin import VPINCalculator
from prediction.price_predictor import PricePredictor

st.set_page_config(
    page_title="Order Book Analysis",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Order Book Analysis + HFT Microstructure")
st.markdown("**LOBSTER Data → Order Book Reconstruction → Microstructure Analysis → Price Prediction**")
st.divider()


# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")

    symbol      = st.selectbox("Symbol", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
    start_price = st.slider("Start Price ($)", 50.0, 500.0, 175.0)
    n_events    = st.slider("Number of Events", 2000, 10000, 5000, step=1000)
    run_btn     = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    st.divider()
    st.markdown("""
    **What this analyzes:**
    1. Simulates LOBSTER order book data
    2. Reconstructs limit order book
    3. Calculates bid-ask spread
    4. Computes Order Flow Imbalance
    5. Measures VPIN (toxicity)
    6. Trains price predictor

    **[View on GitHub](https://github.com/saimanjunathk/order-book-analysis)**
    """)


# ── Run Analysis ──
if run_btn:
    with st.spinner("📊 Generating order book data..."):
        sim    = LOBSTERSimulator(
            symbol=symbol,
            start_price=start_price,
            n_events=n_events
        )
        events = sim.generate_events()

    with st.spinner("🔧 Reconstructing order book..."):
        reconstructor = OrderBookReconstructor(levels=5)
        snapshots     = reconstructor.reconstruct(events)

    with st.spinner("📐 Calculating microstructure features..."):
        spread_analyzer = SpreadAnalyzer(events, snapshots)
        spread_df       = spread_analyzer.quoted_spread()
        intraday_spread = spread_analyzer.intraday_spread("5min")

        ofi_calc = OrderFlowImbalance(events)
        ofi_df   = ofi_calc.calculate_ofi("1min")
        book_imb = ofi_calc.book_imbalance(snapshots)

        vpin_calc = VPINCalculator(n_buckets=50, window=20)
        vpin_df   = vpin_calc.calculate(events)

    with st.spinner("🤖 Training price predictor..."):
        predictor = PricePredictor()
        pred_results = predictor.train(snapshots)

    # Store in session state
    st.session_state["events"]         = events
    st.session_state["snapshots"]      = snapshots
    st.session_state["spread_df"]      = spread_df
    st.session_state["intraday_spread"]= intraday_spread
    st.session_state["ofi_df"]         = ofi_df
    st.session_state["book_imb"]       = book_imb
    st.session_state["vpin_df"]        = vpin_df
    st.session_state["pred_results"]   = pred_results
    st.session_state["symbol"]         = symbol


if "events" not in st.session_state:
    st.info("👈 Configure settings and click **Run Analysis** to start!")
    st.stop()


# ── Load from session state ──
events          = st.session_state["events"]
snapshots       = st.session_state["snapshots"]
spread_df       = st.session_state["spread_df"]
intraday_spread = st.session_state["intraday_spread"]
ofi_df          = st.session_state["ofi_df"]
book_imb        = st.session_state["book_imb"]
vpin_df         = st.session_state["vpin_df"]
pred_results    = st.session_state["pred_results"]
symbol          = st.session_state["symbol"]


# ── KPIs ──
st.subheader("📊 Market Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📋 Total Events", f"{len(events):,}")
with col2:
    avg_spread = spread_df["quoted_spread"].mean() if "quoted_spread" in spread_df.columns else 0
    st.metric("💱 Avg Spread", f"${avg_spread:.4f}")
with col3:
    avg_spread_bps = spread_df["quoted_spread_bps"].mean() if "quoted_spread_bps" in spread_df.columns else 0
    st.metric("📏 Spread (bps)", f"{avg_spread_bps:.2f}")
with col4:
    if pred_results and "accuracy" in pred_results:
        st.metric("🎯 Predictor Accuracy", f"{pred_results['accuracy']*100:.1f}%")
    else:
        st.metric("🎯 Predictor Accuracy", "N/A")

st.divider()


# ── Tabs ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 Order Book",
    "💱 Spread Analysis",
    "🌊 Order Flow",
    "☠️ VPIN",
    "🤖 Price Prediction"
])


# ─────────────────────────────────────────────
# TAB 1: ORDER BOOK
# ─────────────────────────────────────────────
with tab1:
    st.subheader("📖 Order Book Reconstruction")

    # Price chart
    st.markdown("**Mid Price Evolution**")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=snapshots["timestamp"],
        y=snapshots["mid_price"],
        mode="lines",
        name="Mid Price",
        line=dict(color="#00d4ff", width=1)
    ))

    if "best_bid" in snapshots.columns and "best_ask" in snapshots.columns:
        fig.add_trace(go.Scatter(
            x=snapshots["timestamp"],
            y=snapshots["best_bid"],
            mode="lines",
            name="Best Bid",
            line=dict(color="#10b981", width=1, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=snapshots["timestamp"],
            y=snapshots["best_ask"],
            mode="lines",
            name="Best Ask",
            line=dict(color="#ef4444", width=1, dash="dot")
        ))

    fig.update_layout(
        title=f"{symbol} Price Stream",
        xaxis_title="Time",
        yaxis_title="Price ($)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Order book snapshot
    st.markdown("**Latest Order Book Snapshot**")
    latest = snapshots.iloc[-1]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🔴 Ask Side (Sell Orders)**")
        ask_data = []
        for i in range(5, 0, -1):
            if f"ask_price_{i}" in latest and not pd.isna(latest[f"ask_price_{i}"]):
                ask_data.append({
                    "Level": i,
                    "Price": f"${latest[f'ask_price_{i}']:.2f}",
                    "Size":  int(latest[f"ask_size_{i}"])
                })
        if ask_data:
            st.dataframe(pd.DataFrame(ask_data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**🟢 Bid Side (Buy Orders)**")
        bid_data = []
        for i in range(1, 6):
            if f"bid_price_{i}" in latest and not pd.isna(latest[f"bid_price_{i}"]):
                bid_data.append({
                    "Level": i,
                    "Price": f"${latest[f'bid_price_{i}']:.2f}",
                    "Size":  int(latest[f"bid_size_{i}"])
                })
        if bid_data:
            st.dataframe(pd.DataFrame(bid_data), hide_index=True, use_container_width=True)

    # Event distribution
    st.markdown("**Order Event Distribution**")
    event_counts = events["event_type"].value_counts().reset_index()
    event_counts.columns = ["event_type", "count"]
    event_map = {1: "New Order", 2: "Partial Cancel", 3: "Full Cancel",
                 4: "Execution", 5: "Hidden Exec"}
    event_counts["event_name"] = event_counts["event_type"].map(event_map)
    fig = px.pie(event_counts, values="count", names="event_name", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 2: SPREAD ANALYSIS
# ─────────────────────────────────────────────
with tab2:
    st.subheader("💱 Bid-Ask Spread Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Spread Distribution**")
        if "quoted_spread_bps" in spread_df.columns:
            fig = px.histogram(
                spread_df,
                x="quoted_spread_bps",
                nbins=50,
                labels={"quoted_spread_bps": "Spread (bps)"},
                color_discrete_sequence=["#00d4ff"]
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Intraday Spread Pattern**")
        if not intraday_spread.empty:
            fig = px.line(
                intraday_spread,
                x="time",
                y="avg_spread_bps",
                labels={"avg_spread_bps": "Avg Spread (bps)", "time": "Time"},
                color_discrete_sequence=["#7c3aed"]
            )
            st.plotly_chart(fig, use_container_width=True)

    # Spread statistics
    st.markdown("**Spread Statistics**")
    if "quoted_spread" in spread_df.columns:
        stats = {
            "Mean Spread ($)":   f"${spread_df['quoted_spread'].mean():.4f}",
            "Median Spread ($)": f"${spread_df['quoted_spread'].median():.4f}",
            "Mean Spread (bps)": f"{spread_df['quoted_spread_bps'].mean():.2f}",
            "Max Spread ($)":    f"${spread_df['quoted_spread'].max():.4f}",
            "Min Spread ($)":    f"${spread_df['quoted_spread'].min():.4f}",
        }
        stats_df = pd.DataFrame(
            stats.items(),
            columns=["Metric", "Value"]
        )
        st.dataframe(stats_df, hide_index=True, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 3: ORDER FLOW
# ─────────────────────────────────────────────
with tab3:
    st.subheader("🌊 Order Flow Imbalance (OFI)")
    st.caption("OFI measures buying vs selling pressure. Strong predictor of short-term price moves.")

    if not ofi_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**OFI Over Time**")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ofi_df["timestamp"],
                y=ofi_df["ofi_normalized"],
                marker_color=[
                    "#10b981" if v > 0 else "#ef4444"
                    for v in ofi_df["ofi_normalized"]
                ]
            ))
            fig.add_hline(y=0.1,  line_dash="dash", line_color="white")
            fig.add_hline(y=-0.1, line_dash="dash", line_color="white")
            fig.update_layout(
                title="Normalized OFI",
                xaxis_title="Time",
                yaxis_title="OFI"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Buy vs Sell Volume**")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ofi_df["timestamp"],
                y=ofi_df["buy_volume"],
                name="Buy Volume",
                marker_color="#10b981"
            ))
            fig.add_trace(go.Bar(
                x=ofi_df["timestamp"],
                y=-ofi_df["sell_volume"],
                name="Sell Volume",
                marker_color="#ef4444"
            ))
            fig.update_layout(
                barmode="overlay",
                title="Buy vs Sell Volume",
                xaxis_title="Time",
                yaxis_title="Volume"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Signal distribution
        signal_counts = ofi_df["signal"].value_counts().reset_index()
        signal_counts.columns = ["signal", "count"]
        fig = px.pie(
            signal_counts,
            values="count",
            names="signal",
            title="OFI Signal Distribution",
            color_discrete_map={
                "buy_pressure":  "#10b981",
                "sell_pressure": "#ef4444",
                "neutral":       "#6b7280"
            }
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 4: VPIN
# ─────────────────────────────────────────────
with tab4:
    st.subheader("☠️ VPIN — Order Flow Toxicity")
    st.caption("VPIN measures probability of informed trading. High VPIN = dangerous for market makers.")

    if not vpin_df.empty and "vpin" in vpin_df.columns:

        vpin_clean = vpin_df.dropna(subset=["vpin"])

        col1, col2 = st.columns(2)
        with col1:
            avg_vpin = vpin_clean["vpin"].mean()
            st.metric("Average VPIN", f"{avg_vpin:.4f}")
        with col2:
            max_vpin = vpin_clean["vpin"].max()
            st.metric("Max VPIN", f"{max_vpin:.4f}")

        # VPIN over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vpin_clean["timestamp"],
            y=vpin_clean["vpin"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#ef4444", width=2),
            name="VPIN"
        ))
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="orange",
            annotation_text="High Toxicity (0.5)"
        )
        fig.update_layout(
            title="VPIN Over Time",
            xaxis_title="Time",
            yaxis_title="VPIN",
            yaxis_range=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Toxicity distribution
        if "toxicity" in vpin_df.columns:
            tox_counts = vpin_df["toxicity"].value_counts().reset_index()
            tox_counts.columns = ["toxicity", "count"]
            fig = px.bar(
                tox_counts,
                x="toxicity",
                y="count",
                color="toxicity",
                title="VPIN Toxicity Distribution",
                color_discrete_map={
                    "Low":       "#10b981",
                    "Medium":    "#f59e0b",
                    "High":      "#ef4444",
                    "Very High": "#7c3aed"
                }
            )
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# TAB 5: PRICE PREDICTION
# ─────────────────────────────────────────────
with tab5:
    st.subheader("🤖 Short-Horizon Price Prediction")
    st.caption("Uses microstructure features to predict price direction over next 5 ticks")

    if pred_results and "accuracy" in pred_results:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Accuracy",   f"{pred_results['accuracy']*100:.2f}%")
        with col2:
            st.metric("📊 Samples",    f"{pred_results['n_samples']:,}")
        with col3:
            st.metric("🔢 Features",   len(pred_results["importance"]) if "importance" in pred_results else 0)

        if "importance" in pred_results:
            st.markdown("**Feature Importance**")
            st.caption("Which microstructure signals predict price direction best?")
            fig = px.bar(
                pred_results["importance"],
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                labels={"importance": "Importance", "feature": "Feature"}
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        if "report" in pred_results:
            st.markdown("**Classification Report**")
            st.code(pred_results["report"])
    else:
        st.warning("Not enough data for price prediction. Try increasing number of events.")


# ── Sidebar summary ──
with st.sidebar:
    st.divider()
    if "events" in st.session_state:
        st.markdown("**Current Analysis**")
        st.write(f"Symbol: {symbol}")
        st.write(f"Events: {len(events):,}")
        st.write(f"Snapshots: {len(snapshots):,}")