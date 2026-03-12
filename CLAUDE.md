# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires an `ANTHROPIC_API_KEY` in a `.env` file inside `stock_agent/`.

```bash
cd stock_agent
pip install -r requirements.txt
```

## Running

**CLI (interactive loop):**
```bash
cd stock_agent
python main.py
```

**Web server (FastAPI + static UI):**
```bash
cd stock_agent
uvicorn api:app --reload
```
The web UI is served at `/` and the API endpoint is `POST /api/query` with `{"query": "..."}`.

## Architecture

A LangGraph-based stock purchase calculator. Takes natural language input (e.g., "I want to buy Tesla with 3000 shekels") and returns how many shares can be purchased.

**Data flow:** `parse_input` → `convert_currency` → `get_stock_price` → `calculate_shares` → `format_response`

- **`state.py`** — `AgentState` TypedDict: the single shared state object flowing through all nodes
- **`graph.py`** — Builds the `StateGraph`; `_route(next_node)` checks `state["error"]` after each node and short-circuits to `format_response` on any failure
- **`nodes.py`** — Five processing nodes: Claude Sonnet 4.6 parses free-text to JSON; open.er-api.com fetches ILS→USD rate; yfinance fetches stock price
- **`main.py`** — CLI entry point; initializes the full `AgentState` dict before each `graph.invoke()` call
- **`api.py`** — FastAPI server; graph is compiled once at startup via `lifespan`; serves `static/index.html` as the web UI
- **`static/index.html`** — Single-page frontend that calls `POST /api/query`

**Error handling pattern:** Every node returns either its result fields or `{"error": "..."}`. The `_route()` closure in `graph.py` detects errors and bypasses remaining nodes, routing directly to `format_response`.
