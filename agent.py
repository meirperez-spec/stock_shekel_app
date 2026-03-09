import re
import time
from datetime import datetime
from typing import TypedDict, Annotated
import operator

from duckduckgo_search import DDGS
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from stock_data import fetch_stock_data, format_stock_summary
from config import ANTHROPIC_API_KEY, STOCKS, CHECK_INTERVAL


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    stocks: list[str]
    stock_data: list[dict]
    analyses: list[dict]
    run_count: int
    last_run: str


# ── LLM setup ─────────────────────────────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    api_key=ANTHROPIC_API_KEY,
    max_tokens=2048,
)

QUESTION_PROMPT = """You are a financial analyst assistant. You will be given current market data for a stock, recent web search results, and a specific question. Answer using the provided data — do not rely on prior training knowledge for facts."""

COMPARE_PROMPT = """You are a financial analyst assistant. You will be given market data for multiple stocks and recent web search results. Provide a structured comparison covering valuation, performance, growth, and risk. End with a clear conclusion. Use only the provided data — do not rely on prior training knowledge for facts."""

GENERAL_PROMPT = """You are a financial analyst assistant. You will be given recent web search results and a question. Answer using only the provided search results — do not rely on prior training knowledge for facts. If the search results don't contain enough information, say so."""

SYSTEM_PROMPT = """You are a financial analyst assistant. You will be given current market data for a stock and must provide a concise trading recommendation.

For each stock, respond with EXACTLY this format:
RECOMMENDATION: [BUY / SELL / HOLD]
CONFIDENCE: [HIGH / MEDIUM / LOW]
REASONING: <2-3 sentences explaining the key factors driving your recommendation>

Base your analysis on:
- Price vs moving averages (trend direction)
- RSI (overbought >70, oversold <30)
- Price position relative to 52-week range
- Volume relative to average (confirms moves)
- P/E ratio context
- Day change momentum

Be direct and actionable. Do not add disclaimers about seeking financial advice."""


# ── Nodes ─────────────────────────────────────────────────────────────────────

def fetch_prices(state: AgentState) -> AgentState:
    print(f"\n{'='*60}")
    print(f"  Stock Analysis Run #{state['run_count'] + 1}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"\nFetching data for: {', '.join(state['stocks'])} ...")

    data = []
    for ticker in state["stocks"]:
        try:
            d = fetch_stock_data(ticker)
            data.append(d)
            status = f"${d['current_price']} ({d['day_change_pct']:+.2f}%)" if "error" not in d else f"ERROR: {d['error']}"
            print(f"  {ticker}: {status}")
        except Exception as e:
            data.append({"ticker": ticker, "error": str(e)})
            print(f"  {ticker}: ERROR - {e}")

    return {**state, "stock_data": data}


def analyze_stocks(state: AgentState) -> AgentState:
    print("\nAnalyzing stocks with Claude...\n")
    analyses = []

    for stock in state["stock_data"]:
        if "error" in stock:
            analyses.append({
                "ticker": stock["ticker"],
                "recommendation": "N/A",
                "confidence": "N/A",
                "reasoning": f"Data fetch failed: {stock['error']}",
            })
            continue

        summary = format_stock_summary(stock)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Analyze this stock:\n\n{summary}"),
        ]

        try:
            response = llm.invoke(messages)
            raw = response.content.strip()

            # Parse the structured response
            rec = _parse_field(raw, "RECOMMENDATION")
            conf = _parse_field(raw, "CONFIDENCE")
            reason = _parse_field(raw, "REASONING")

            analyses.append({
                "ticker": stock["ticker"],
                "company_name": stock.get("company_name", stock["ticker"]),
                "current_price": stock["current_price"],
                "day_change_pct": stock["day_change_pct"],
                "recommendation": rec,
                "confidence": conf,
                "reasoning": reason,
            })
        except Exception as e:
            analyses.append({
                "ticker": stock["ticker"],
                "recommendation": "ERROR",
                "confidence": "N/A",
                "reasoning": str(e),
            })

    return {**state, "analyses": analyses}


def report(state: AgentState) -> AgentState:
    print(f"\n{'-'*60}")
    print("  ANALYSIS RESULTS")
    print(f"{'-'*60}\n")

    for a in state["analyses"]:
        rec = a["recommendation"]
        color = _rec_color(rec)
        conf = a.get("confidence", "N/A")
        price = a.get("current_price", "N/A")
        change = a.get("day_change_pct", 0)
        name = a.get("company_name", a["ticker"])

        print(f"  {a['ticker']} — {name}")
        print(f"  Price: ${price} ({change:+.2f}%)" if isinstance(change, float) else f"  Price: ${price}")
        print(f"  {color}  Confidence: {conf}")
        print(f"  {a['reasoning']}")
        print()

    print(f"{'-'*60}")
    print(f"  Next check in {_format_interval(CHECK_INTERVAL)}")
    print(f"{'-'*60}\n")

    return {
        **state,
        "run_count": state["run_count"] + 1,
        "last_run": datetime.now().isoformat(),
    }


def ask_user(state: AgentState) -> AgentState:
    while True:
        try:
            user_input = input("\nAsk anything (e.g. 'NVDA', 'NVDA compare valuation to peers', 'what is a P/E ratio?'), or press Enter to continue: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            break
        comparison = _extract_comparison(user_input)
        if comparison:
            tickers, question = comparison
            compare_stocks(tickers, question)
        else:
            ticker, question = _extract_ticker(user_input)
            if ticker:
                query_stock(ticker, question=question)
            else:
                ask_general(user_input)
    return state


def wait_node(state: AgentState) -> AgentState:
    time.sleep(CHECK_INTERVAL)
    return state


def should_continue(state: AgentState) -> str:
    return "fetch_prices"  # loop forever


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_field(text: str, field: str) -> str:
    for line in text.splitlines():
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()
    return "N/A"


def _rec_color(rec: str) -> str:
    icons = {"BUY": ">>> BUY <<<", "SELL": ">>> SELL <<<", "HOLD": "--- HOLD ---"}
    return icons.get(rec.upper(), rec)


def _extract_ticker(text: str) -> tuple[str, str]:
    """Extract the first word as a potential ticker/company name, plus the remainder."""
    m = re.match(r'^(\S+)(.*)', text)
    if m:
        return m.group(1), m.group(2).strip()
    return "", text


def _extract_comparison(text: str) -> tuple[list[str], str] | None:
    """Detect comparison queries. Returns (raw_tickers, question) or None.

    Supported patterns:
      X vs Y [question]
      X versus Y [question]
      compare X Y [Z ...] [question]
    """
    # Pattern: "X vs Y [question]"
    m = re.match(r'^(\S+)\s+(?:vs\.?|versus)\s+(\S+)(.*)', text, re.IGNORECASE)
    if m:
        return [m.group(1), m.group(2)], m.group(3).strip()

    # Pattern: "compare X Y [Z ...] [question]"
    m = re.match(r'^compare\s+(.*)', text, re.IGNORECASE)
    if m:
        tokens = m.group(1).split()
        tickers, question_words = [], []
        for i, token in enumerate(tokens):
            if not question_words and re.match(r'^[A-Za-z.]{1,10}$', token):
                tickers.append(token)
            else:
                question_words = tokens[i:]
                break
        if len(tickers) >= 2:
            return tickers, " ".join(question_words)

    return None


def _resolve_ticker(query: str) -> str:
    """Use LLM to resolve a company name or fuzzy ticker to a valid ticker symbol. Returns '' if not a stock."""
    try:
        response = llm.invoke([
            SystemMessage(content="You are a financial assistant. If the input refers to a publicly traded company or stock, respond with only its ticker symbol in uppercase (e.g. NVDA). If it does not refer to a specific stock, respond with only NONE."),
            HumanMessage(content=query),
        ])
        result = response.content.strip().upper()
        if re.match(r'^[A-Z]{1,6}(\.[A-Z])?$', result) and result != "NONE":
            return result
    except Exception:
        pass
    return ""


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return formatted results."""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"[{r['title']}] {r['body']}")
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


def ask_general(question: str) -> None:
    """Answer a freeform financial question using live web search results."""
    print("  Searching the web...")
    search_results = web_search(question)
    try:
        response = llm.invoke([
            SystemMessage(content=GENERAL_PROMPT),
            HumanMessage(content=f"Search results:\n\n{search_results}\n\nQuestion: {question}"),
        ])
        print(f"\n{'-'*60}")
        print(f"  {response.content.strip()}")
        print(f"{'-'*60}\n")
    except Exception as e:
        print(f"ERROR: {e}")


def _format_interval(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m" if seconds % 3600 else f"{seconds // 3600}h"


# ── On-demand query ───────────────────────────────────────────────────────────

def query_stock(ticker: str, question: str = "") -> None:
    """Fetch and analyze a single stock on demand. If question is provided, answer it directly."""
    original_input = f"{ticker} {question}".strip()
    ticker = ticker.upper()
    print(f"\nQuerying {ticker}...")

    try:
        data = fetch_stock_data(ticker)
    except Exception as e:
        data = {"ticker": ticker, "error": str(e)}

    if "error" in data:
        resolved = _resolve_ticker(ticker)
        if resolved and resolved != ticker:
            print(f"  Resolved '{ticker}' → {resolved}")
            ticker = resolved
            try:
                data = fetch_stock_data(ticker)
            except Exception as e:
                data = {"ticker": ticker, "error": str(e)}

    if "error" in data:
        print(f"  Could not find stock data, answering from general knowledge...")
        ask_general(original_input)
        return

    summary = format_stock_summary(data)
    name = data.get("company_name", ticker)
    price = data["current_price"]
    change = data["day_change_pct"]

    if question:
        print("  Searching the web...")
        search_results = web_search(f"{ticker} {question}")
        messages = [
            SystemMessage(content=QUESTION_PROMPT),
            HumanMessage(content=f"Stock data:\n\n{summary}\n\nWeb search results:\n\n{search_results}\n\nQuestion: {question}"),
        ]
        try:
            response = llm.invoke(messages)
            print(f"\n{'-'*60}")
            print(f"  {ticker} — {name}  |  ${price} ({change:+.2f}%)")
            print(f"\n  Q: {question}")
            print(f"\n  {response.content.strip()}")
            print(f"{'-'*60}\n")
        except Exception as e:
            print(f"ERROR: Analysis failed: {e}")
    else:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Analyze this stock:\n\n{summary}"),
        ]
        try:
            response = llm.invoke(messages)
            raw = response.content.strip()

            rec = _parse_field(raw, "RECOMMENDATION")
            conf = _parse_field(raw, "CONFIDENCE")
            reason = _parse_field(raw, "REASONING")

            print(f"\n{'-'*60}")
            print(f"  {ticker} — {name}")
            print(f"  Price: ${price} ({change:+.2f}%)")
            print(f"  {_rec_color(rec)}  Confidence: {conf}")
            print(f"  {reason}")
            print(f"{'-'*60}\n")
        except Exception as e:
            print(f"ERROR: Analysis failed: {e}")


def compare_stocks(raw_tickers: list[str], question: str = "") -> None:
    """Fetch data for multiple stocks and produce a structured comparison."""
    print(f"\nComparing {', '.join(t.upper() for t in raw_tickers)}...")

    summaries, resolved = [], []
    for raw in raw_tickers:
        ticker = raw.upper()
        try:
            data = fetch_stock_data(ticker)
        except Exception as e:
            data = {"ticker": ticker, "error": str(e)}

        if "error" in data:
            r = _resolve_ticker(ticker)
            if r and r != ticker:
                print(f"  Resolved '{ticker}' → {r}")
                ticker = r
                try:
                    data = fetch_stock_data(ticker)
                except Exception as e:
                    data = {"ticker": ticker, "error": str(e)}

        if "error" in data:
            print(f"  WARNING: Could not fetch data for {ticker}, skipping.")
            continue

        resolved.append(ticker)
        name = data.get("company_name", ticker)
        price = data["current_price"]
        change = data["day_change_pct"]
        print(f"  {ticker} ({name}): ${price} ({change:+.2f}%)")
        summaries.append(format_stock_summary(data))

    if len(resolved) < 2:
        print("  ERROR: Need at least 2 valid stocks to compare.")
        return

    search_query = f"{' vs '.join(resolved)} comparison {question}".strip()
    print("  Searching the web...")
    search_results = web_search(search_query)

    prompt = question or f"Compare {' vs '.join(resolved)}: valuation, performance, growth, and risks."
    combined = "\n\n---\n\n".join(summaries)

    try:
        response = llm.invoke([
            SystemMessage(content=COMPARE_PROMPT),
            HumanMessage(content=f"Stock data:\n\n{combined}\n\nWeb search results:\n\n{search_results}\n\nQuestion: {prompt}"),
        ])
        print(f"\n{'-'*60}")
        print(f"  Comparison: {' vs '.join(resolved)}")
        print(f"\n{response.content.strip()}")
        print(f"{'-'*60}\n")
    except Exception as e:
        print(f"ERROR: Comparison failed: {e}")


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("fetch_prices", fetch_prices)
    graph.add_node("analyze_stocks", analyze_stocks)
    graph.add_node("report", report)
    graph.add_node("ask_user", ask_user)
    graph.add_node("wait", wait_node)

    graph.set_entry_point("fetch_prices")
    graph.add_edge("fetch_prices", "analyze_stocks")
    graph.add_edge("analyze_stocks", "report")
    graph.add_edge("report", "ask_user")
    graph.add_edge("ask_user", "wait")
    graph.add_conditional_edges("wait", should_continue, {"fetch_prices": "fetch_prices"})

    return graph.compile()
