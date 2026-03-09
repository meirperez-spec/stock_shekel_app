# Stock Trading Agent

An AI-powered stock analysis agent built with [LangGraph](https://github.com/langchain-ai/langgraph) and Claude. It monitors a configurable list of tickers on a schedule, delivers BUY/SELL/HOLD recommendations, and supports interactive freeform Q&A backed by live market data and web search.

## Features

- **Scheduled monitoring** — fetches and analyses all configured tickers on a set interval
- **BUY / SELL / HOLD recommendations** — Claude reasons over price, SMA20/50, RSI-14, 52-week range, volume, and P/E
- **Interactive Q&A** — after each report, ask anything: ticker lookup, specific questions, or general market queries
- **Fuzzy ticker resolution** — type `NVIDIA` and the agent resolves it to `NVDA` automatically
- **Stock comparison** — `NVDA vs AMD` or `compare NVDA AMD MSFT` for a structured side-by-side analysis
- **Live web search** — all answers are grounded in real-time DuckDuckGo results, not stale training data
- **On-demand CLI query** — `python main.py AAPL` for a one-off analysis without starting the loop

## Architecture

```
fetch_prices → analyze_stocks → report → ask_user → wait → (repeat)
```

| File | Role |
|---|---|
| `main.py` | Entry point; routes CLI args or starts the LangGraph loop |
| `agent.py` | Graph nodes, prompts, interactive Q&A, comparison, web search |
| `stock_data.py` | yfinance fetching + indicator calculation (RSI, SMA, P/E, …) |
| `config.py` | Loads `.env` variables |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
cp .env .env.backup   # already configured — keep .env out of git
python main.py
```

## Configuration

Copy `.env` and fill in your values:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `STOCKS` | `AAPL,GOOGL,MSFT,TSLA,AMZN` | Comma-separated tickers to monitor |
| `CHECK_INTERVAL` | `3600` | Seconds between analysis runs |

## Usage

### Continuous monitoring loop

```bash
python main.py
```

### One-off ticker query

```bash
python main.py NVDA
```

### Interactive Q&A (after each report)

```
Ask anything (e.g. 'NVDA', 'NVDA compare valuation to peers', 'what is a P/E ratio?'), or press Enter to continue:

  NVDA                              # standard BUY/SELL/HOLD analysis
  NVIDIA                            # fuzzy match → resolves to NVDA
  NVDA what is the revenue growth?  # specific question with live data + web search
  NVDA vs AMD                       # side-by-side comparison
  compare NVDA AMD MSFT             # multi-stock comparison
  what is a P/E ratio?              # general question answered via web search
```

## Requirements

- Python 3.11+
- Anthropic API key
