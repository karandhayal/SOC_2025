**Overview:** Equity Research involves evaluating publically trading companies to assist investors to take informed decisions. Traditionally its been manual and time intensive task and this project aims to automate the
significant part of this process by integrating:
1. Fundamental Analysis
2. Technical Analysis
3. Extracting financial data
4. Plotting the graphs

**Libraris Used:**
1. NumPy: It is a foundational library used for numerical computation e.g. diff(),mean(),std(),etc.

Sample Code snippet:
```
import numpy as np
prices = np.array([100, 102, 105, 103])
returns = np.diff(prices) / prices[:-1]
```
2. Pandas: Used to handle data by cleaning, strurcturing and analysing data.

Sample Code Snippet
```
df = yf.download(company, start='2020-04-01', end='2025-04-01')
if isinstance(df.columns, pd.MultiIndex):
  df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
lookback = 15
df['UpperBand'] = df['High'].shift(1).rolling(window=lookback).max()
df['LowerBand'] = df['Low'].shift(1).rolling(window=lookback).min()
```
3. YFinance: Used to extract financial data
```
companies = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS']
for company in companies:
    df = yf.download(company, start='2020-04-01', end='2025-04-01')
```
4. MatPlotLib: Matplotlib is used to visualize stock price trends and overlay technical indicators.
```
import matplotlib.pyplot as plt

plt.plot(data['Close'])
plt.title("TCS Stock Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()
```
**Fundamental Analysis**

Sourced from Zerodha Varsity, fundamental analysis evaluates a company’s intrinsic value and financial health. The metrics used are as follows:

**Profitability Ratios**

ROCE = 	EBIT / Capital Employed	Measures return on capital used

ROE	= Net Income / Equity	Shareholder return indicator

Operating Margin =	EBIT / Revenue	Measures operational efficiency

Net Profit Margin =	Net Profit / Revenue	Shows bottom-line profitability

**Valuation Ratios**

P/E Ratio =	Market Price / EPS	Indicates market expectation

P/B Ratio =	Market Price / Book Value	Compares price to asset value

EV/EBITDA =	(EV = Market Cap + Debt – Cash) / EBITDA	Comprehensive valuation metric

**Leverage & Solvency Ratios**

Debt-to-Equity =	Total Debt / Shareholders’ Equity	Measures leverage risk

Interest Coverage =	EBIT / Interest	Debt servicing capability

**Liquidity Ratios**

Current Ratio =	Current Assets / Current Liabilities	Short-term liquidity check

Quick Ratio =	(Current Assets - Inventory) / Liabilities	Liquidity under stress

**Efficiency Ratios**

Inventory Turnover =	COGS / Avg. Inventory	Operational efficiency

Asset Turnover =	Revenue / Total Assets	Efficiency in asset usage

**Technical Analysis:**
Technical analysis evaluates price patterns using historical data and indicators.

**Exponential Moving Average(EMA)**:
Smooths out price action and gives more weight to recent prices, making it more responsive than Simple Moving Average (SMA). Used to identify trends and dynamic support/resistance levels.

Mathematics:

EMA_t = (P_t * K) + (EMA_(t-1) * (1 - K)), where K = 2 / (n + 1)
```
def calculate_ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()
```

**Moving Average Convergence Divergence(MACD):**
Purpose: Identifies trend direction and potential reversals.

Mathematics:

generally,
MACD = EMA_12-EMA_26, Singal = EMA_9

```
def calculate_macd(prices, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line
```

**Relative Strength Index(RSI):**

Purpose: Measures momentum to detect overbought or oversold conditions.

RS = avg gains/avg loss, RSI = 100*(RS/(1+RS))

```
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
```
Bollinger Bands
Purpose:
Measures price volatility. It consists of a middle SMA line, and upper/lower bands that are a fixed number of standard deviations away from the mean.

Mathematics:

Middle Band = SMA_20
Upper Band = SMA_20 + 2*std_deviation
Lower Band = SMA_20 - 2*std_deviation

**Use Cases:**
1. Identifying overbought/oversold zones
2. Volatility analysis
3. Squeeze strategy: band contraction followed by breakout

Plan for the other half:
1. Build user friendly working python program to analyse equity
2. Extend the scope to of this project to portfolio optimization

Resources Used:
Varsity Modules,
Binance Academy,
GitHub Repo provided by the mentor,
mlcourse.ai


**After Mid-term report**

**Part 1: Integrated Fundamental Screening**

The first step in the final project was to translate the fundamental analysis concepts into a robust filtering mechanism. The script fetches key financial ratios for all NIFTY 50 stocks using the yfinance library. It then applies a strict filter to automatically screen for companies that exhibit strong financial health, ensuring that only high-quality businesses are considered for further analysis.
```
def get_fundamentals(ticker):
    """
    Fetches key fundamental metrics for a given stock ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        required_keys = ['trailingPE', 'returnOnEquity', 'returnOnAssets', 'revenueGrowth']
        if not all(key in info and info[key] is not None for key in required_keys):
            return None
        return {
            "Ticker": ticker,
            "PE": info.get("trailingPE"),
            "ROE": info.get("returnOnEquity") * 100,
            "ROA": info.get("returnOnAssets") * 100,
            "CAGR": info.get("revenueGrowth") * 100,
        }
    except Exception:
        return None

# Apply our strict filtering criteria
filtered_stocks = df_fund[
    (df_fund["PE"] < 40) &
    (df_fund["ROE"] > 15) &
    (df_fund["ROA"] > 5)
].copy()
```
**Part 2: Scoring and Ranking Engine**

After filtering, the project needed a way to rank the remaining stocks objectively. A weighted scoring engine was developed for this purpose. Each fundamental and technical metric is normalized to a common scale (0-100) to allow for fair comparison. These individual scores are then combined into a Total_Score, with higher weights given to metrics like P/E ratio and ROE, which are strong indicators of value and profitability. This provides a clear, data-driven ranking of the top investment opportunities.

```
def normalize(series, inverse=False):
    """Normalizes a pandas series to a 0-100 scale for fair comparison."""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val: return pd.Series(100, index=series.index)
    if inverse:
        return 100 * (max_val - series) / (max_val - min_val)
    else:
        return 100 * (series - min_val) / (max_val - min_val)

# Calculate the final, weighted Total Score.
final_df["Total_Score"] = (
    final_df["score_pe"] * 0.25 +
    final_df["score_roe"] * 0.25 +
    final_df["score_roa"] * 0.20 +
    final_df["score_cagr"] * 0.20 +
    final_df["score_rsi"] * 0.10
)

# Sort by the final score to get our top recommendations.
recommendations = final_df.sort_values("Total_Score", ascending=False)
```
**Part 3: Strategy Backtesting**

The final and most critical part of the project was to validate the strategy. A comprehensive backtester was built to simulate trading the top 5 ranked stocks over a 5-year period. This section integrates the technical analysis concepts into a practical trading model.

**Strategy Logic**

The trading strategy is a multi-condition model that combines momentum, trend, and breakout indicators to generate signals:

 **Buy Signals**: A position is initiated if any of these conditions are met:

1. Breakout Confirmation: The price closes above the 15-day Upper Band with a slight margin, while RSI is below 75 and MACD is positive.

2. Volume Retest: The price breaks the Upper Band, confirmed by a spike in volume (higher than the 30-day average).

3. Deeply Oversold: The RSI drops below 14, indicating a potential bounce.

**Sell Signals**: A position is closed if either of these conditions are met:

1. The price breaks below the 15-day Lower Band.

2. The MACD line crosses below its signal line, indicating weakening momentum.

```
# Define Buy/Sell Signals based on the reference logic
entry_cond = (
    (df['Close'] > df['UpperBand'] * 1.005) &
    (df['RSI'] < 75) &
    (df['MACD'] > df['MACD_Signal'])
)
retest_cond = (
    (df['Close'] > df['UpperBand']) &
    (df['Volume'] > df['Volume'].rolling(30).mean())
)
df['Signal'] = 0
df.loc[(entry_cond) | (df['RSI'] < 14) | (retest_cond), 'Signal'] = 1

exit_cond = (
    (df['Close'] < df['LowerBand'] * 0.995) |
    (df['MACD'] < df['MACD_Signal'] * 0.995)
)
df.loc[exit_cond, 'Signal'] = -1
```
**Risk Management & Portfolio Optimization**

To make the backtest realistic, a sophisticated risk management and dynamic allocation model was included:

1. **Dynamic Cash Allocation:** Instead of assigning a fixed amount to each stock, the available cash is dynamically divided among all open positions and new buy signals on any given day. This ensures capital is deployed efficiently.

2. **Stop-Loss:** A hard 5% stop-loss is implemented on every trade to limit downside risk.

3. **Take-Profit:** A position is automatically sold if it reaches an 8% profit to lock in gains.

4. **Trailing Stop-Loss:** A 3% trailing stop-loss is used to protect profits. As the stock price rises, the stop-loss level also rises, securing a portion of the gains while allowing for further upside.

```
# Check for exit conditions
if (current_price <= trailing_stop or 
    current_price <= entry_price * (1 - stop_loss_pct) or
    current_price >= entry_price * (1 + take_profit_pct) or
    row['Signal'] == -1):
    cash += shares * current_price
    del positions[ticker]
```
**Interpreting the Results**

The script outputs three key results:

1. The number of stocks that passed the fundamental screening.

2. A ranked list of the top 5 recommended stocks with their key metrics and total score.

3. The final backtesting results, including the Compound Annual Growth Rate (CAGR), which measures the strategy's annualized return.

**Future Scope**

This project provides a strong foundation that can be expanded in several ways:

1. Incorporate More Metrics: Add other fundamental ratios like Debt-to-Equity or technical indicators like Bollinger Bands.

2. Machine Learning: Implement a machine learning model to predict buy/sell signals based on historical patterns.

3. Expand to Other Markets: Adapt the script to analyze international stocks, ETFs, or other asset classes.

4. Advanced Portfolio Optimization: Use techniques like the Sharpe Ratio or Modern Portfolio Theory to build an optimally balanced portfolio.

