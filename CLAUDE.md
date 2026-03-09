# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Agent

```bash
pip install -r requirements.txt
cp .env.txt .env   # fill in ANTHROPIC_API_KEY, then add .env to .gitignore
python main.py
```

The agent runs in an infinite loop. Stop with `Ctrl+C`.

## Configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Anthropic API key |
| `STOCKS` | `AAPL,GOOGL,MSFT,TSLA,AMZN` | Comma-separated tickers |
| `CHECK_INTERVAL` | `3600` | Seconds between analysis runs |

## Architecture

**LangGraph loop:** `fetch_prices → analyze_stocks → report → wait → (repeat)`

- **`config.py`** — Loads env vars; imported by all other modules.
- **`stock_data.py`** — `fetch_stock_data(ticker)` calls yfinance for price, SMA20/50, RSI-14, 52W range, volume, P/E. `format_stock_summary()` renders it as a plain-text string for the LLM prompt.
- **`agent.py`** — Defines `AgentState` (TypedDict), four graph nodes, and `build_graph()`. The LLM (`claude-sonnet-4-6`) is instantiated at module level. Claude's response must follow a strict `RECOMMENDATION / CONFIDENCE / REASONING` format; `_parse_field()` extracts each line.
- **`main.py`** — Validates API key, constructs initial state, calls `graph.invoke()`.

## Key Patterns

- **State immutability:** Each node returns `{**state, ...updated_fields}` — never mutates state in place.
- **Error handling per ticker:** Errors from yfinance are caught per-ticker and stored as `{"ticker": ..., "error": ...}` dicts; downstream nodes check for the `"error"` key and skip LLM calls for failed tickers.
- **LLM response parsing:** `_parse_field(raw, "FIELD")` scans lines for `FIELD:` prefix — fragile to format deviations. If Claude doesn't follow the exact format, fields return `"N/A"`.
