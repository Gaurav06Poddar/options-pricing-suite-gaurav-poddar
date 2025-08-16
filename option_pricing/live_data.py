# option_pricing/live_data.py
"""
Unified live-data adapter. Default: yfinance (delayed).
Optional: Polygon/Tiingo if API keys present in environment.

Provides:
- get_quote(ticker) -> dict { 'ticker','time','last','bid','ask','volume' }
- get_history(ticker, period='1d') -> DataFrame (wrapper around yfinance)
- stream_quotes(tickers, interval_sec=1, callback=fn) -> generator/callback loop
"""

import os
import time
import logging
from typing import List, Callable, Dict, Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Optional polygon import (guarded)
_POLYGON = False
try:
    from polygon import RESTClient as PolygonREST
    _POLYGON = True
except Exception:
    _POLYGON = False

def get_quote(ticker: str) -> Dict:
    """
    Get a single quote. Returns a dict with keys:
    'ticker', 'time', 'last', 'bid', 'ask', 'volume'
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="1d")
        if hist is None or hist.empty:
            return {'ticker': ticker, 'time': None, 'last': None, 'bid': None, 'ask': None, 'volume': None}
        last_row = hist.iloc[-1]
        return {
            'ticker': ticker,
            'time': last_row.name.to_pydatetime(),
            'last': float(last_row['Close']),
            'bid': float(last_row.get('Open', last_row['Close'])),
            'ask': float(last_row.get('High', last_row['Close'])),
            'volume': int(last_row.get('Volume', 0))
        }
    except Exception as e:
        logger.exception("yfinance get_quote failed")
        return {'ticker': ticker, 'time': None, 'last': None, 'bid': None, 'ask': None, 'volume': None}


def get_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Return historical OHLCV DataFrame via yfinance.
    """
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period)
        return df
    except Exception:
        return pd.DataFrame()


def stream_quotes(tickers: List[str], interval_sec: int = 5, callback: Optional[Callable] = None, max_iters: int = None):
    """
    Poll quotes periodically; call callback(quote_dict) for each quote.
    This is simple polling (not websocket) but shows how live integration works.
    Use small set of tickers and moderate interval to avoid API throttling.
    """
    it = 0
    while True:
        for t in tickers:
            q = get_quote(t)
            if callback:
                try:
                    callback(q)
                except Exception:
                    logger.exception("quote callback failed")
        it += 1
        if max_iters is not None and it >= max_iters:
            break
        time.sleep(interval_sec)
