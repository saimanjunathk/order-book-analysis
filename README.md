# Order Book Analysis + HFT Microstructure

## Live Demo
Coming soon on Streamlit Cloud

## Architecture
LOBSTER Simulator -> Order Book Reconstruction -> Microstructure Features -> Price Prediction

## Features
- Order Book Reconstruction: Full limit order book tick by tick
- Bid-Ask Spread Analysis: Quoted spread, effective spread, intraday patterns
- Order Flow Imbalance: Buy vs sell pressure measurement
- VPIN: Order flow toxicity measurement
- Price Prediction: GradientBoosting on microstructure features

## Tech Stack
- Data: LOBSTER-style simulation
- Processing: Python, Pandas, NumPy
- ML: Scikit-learn GradientBoosting
- Dashboard: Streamlit + Plotly

## How to Run Locally
git clone https://github.com/saimanjunathk/order-book-analysis
cd order-book-analysis
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run dashboard/app.py

## Status
- LOBSTER Simulator - Done
- Order Book Reconstruction - Done
- Spread Analysis - Done
- Order Flow Imbalance - Done
- VPIN Calculator - Done
- Price Predictor - Done
- Live Dashboard - Done