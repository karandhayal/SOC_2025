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
