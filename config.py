import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
STOCKS = [s.strip() for s in os.getenv("STOCKS", "AAPL,GOOGL,MSFT,TSLA,AMZN").split(",")]
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "3600"))
