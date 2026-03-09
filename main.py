import sys
from config import STOCKS, CHECK_INTERVAL, ANTHROPIC_API_KEY
from agent import build_graph, AgentState, query_stock


def main():
    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    if len(sys.argv) > 1:
        query_stock(sys.argv[1])
        return

    print("Stock Trading Agent")
    print(f"Monitoring: {', '.join(STOCKS)}")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print("Press Ctrl+C to stop.\n")

    graph = build_graph()

    initial_state: AgentState = {
        "stocks": STOCKS,
        "stock_data": [],
        "analyses": [],
        "run_count": 0,
        "last_run": "",
    }

    try:
        graph.invoke(initial_state)
    except KeyboardInterrupt:
        print("\nAgent stopped.")


if __name__ == "__main__":
    main()
