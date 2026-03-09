import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str) -> dict:
    """Fetch current price and technical indicators for a stock."""
    stock = yf.Ticker(ticker)

    # Get 3 months of daily history for indicator calculation
    hist = stock.history(period="3mo")
    if hist.empty:
        return {"ticker": ticker, "error": "No data available"}

    close = hist["Close"]
    current_price = round(float(close.iloc[-1]), 2)

    # Simple Moving Averages
    sma_20 = round(float(close.rolling(20).mean().iloc[-1]), 2)
    sma_50 = round(float(close.rolling(50).mean().iloc[-1]), 2) if len(close) >= 50 else None

    # RSI (14-period)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    rsi = round(float((100 - 100 / (1 + rs)).iloc[-1]), 2)

    # Price change
    prev_close = round(float(close.iloc[-2]), 2)
    day_change_pct = round((current_price - prev_close) / prev_close * 100, 2)

    # 52-week high/low
    hist_1y = stock.history(period="1y")
    week_52_high = round(float(hist_1y["High"].max()), 2) if not hist_1y.empty else None
    week_52_low = round(float(hist_1y["Low"].min()), 2) if not hist_1y.empty else None

    # Volume vs avg volume
    avg_volume = int(hist["Volume"].rolling(20).mean().iloc[-1])
    current_volume = int(hist["Volume"].iloc[-1])

    info = stock.info
    pe_ratio = info.get("trailingPE")
    market_cap = info.get("marketCap")
    company_name = info.get("shortName", ticker)

    return {
        "ticker": ticker,
        "company_name": company_name,
        "current_price": current_price,
        "prev_close": prev_close,
        "day_change_pct": day_change_pct,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "rsi_14": rsi,
        "week_52_high": week_52_high,
        "week_52_low": week_52_low,
        "current_volume": current_volume,
        "avg_volume_20d": avg_volume,
        "pe_ratio": round(pe_ratio, 2) if pe_ratio else None,
        "market_cap": market_cap,
    }


def format_stock_summary(data: dict) -> str:
    """Format stock data into a readable string for the LLM."""
    if "error" in data:
        return f"{data['ticker']}: ERROR - {data['error']}"

    mc = data["market_cap"]
    mc_str = f"${mc / 1e9:.1f}B" if mc and mc >= 1e9 else (f"${mc / 1e6:.1f}M" if mc else "N/A")

    sma50_str = f"${data['sma_50']}" if data['sma_50'] else 'N/A'
    return (
        f"Ticker: {data['ticker']} ({data['company_name']})\n"
        f"  Price: ${data['current_price']} (Day change: {data['day_change_pct']:+.2f}%)\n"
        f"  SMA20: ${data['sma_20']}  |  SMA50: {sma50_str}\n"
        f"  RSI(14): {data['rsi_14']}\n"
        f"  52W High: ${data['week_52_high']}  |  52W Low: ${data['week_52_low']}\n"
        f"  Volume: {data['current_volume']:,} (20d avg: {data['avg_volume_20d']:,})\n"
        f"  P/E: {data['pe_ratio'] if data['pe_ratio'] else 'N/A'}  |  Market Cap: {mc_str}"
    )
