# option_pricing/adapters.py
"""
Market data adapters: unified interface for yfinance and optional Polygon/Tiingo.

Usage:
  from option_pricing.adapters import MarketAdapter
  ma = MarketAdapter()
  q = ma.get_quote('AAPL')

Behavior:
 - If POLYGON_API_KEY or TIINGO_API_KEY present, try to use them (Polygon preferred).
 - Otherwise use yfinance (delayed but free).
"""

import os
import logging
from typing import Dict, Optional, List
import pandas as pd
import datetime

logger = logging.getLogger(__name__)

# yfinance fallback
import yfinance as yf

# optional polygon client
_POLYGON = False
try:
    from polygon import RESTClient as PolygonREST
    _POLYGON = True
except Exception:
    _POLYGON = False

_TIINGO = False
try:
    import tiingo
    _TIINGO = True
except Exception:
    _TIINGO = False


class MarketAdapter:
    def __init__(self, polygon_key: Optional[str] = None, tiingo_key: Optional[str] = None):
        self.polygon_key = polygon_key or os.environ.get("POLYGON_API_KEY")
        self.tiingo_key = tiingo_key or os.environ.get("TIINGO_API_KEY")
        self._use_polygon = _POLYGON and bool(self.polygon_key)
        self._use_tiingo = _TIINGO and bool(self.tiingo_key) and not self._use_polygon
        if self._use_polygon:
            self._client = PolygonREST(self.polygon_key)
        elif self._use_tiingo:
            from tiingo import TiingoClient
            config = {"api_key": self.tiingo_key}
            self._client = TiingoClient(config)
        else:
            self._client = None

    def get_quote(self, ticker: str) -> Dict:
        ticker = ticker.upper()
        if self._use_polygon:
            try:
                agg = self._client.get_last_trade(ticker)
                # Polygon returns dict-like object
                return {
                    "ticker": ticker,
                    "time": datetime.datetime.utcfromtimestamp(agg['t'] / 1000.0),
                    "last": float(agg['p']),
                    "bid": None,
                    "ask": None,
                    "volume": None
                }
            except Exception:
                logger.exception("Polygon get_quote failed, falling back to yfinance")
        if self._use_tiingo:
            try:
                res = self._client.get_latest_price(ticker, fmt='json')
                if isinstance(res, list):
                    price = res[0] if res else None
                else:
                    price = res
                return {"ticker": ticker, "time": datetime.datetime.utcnow(), "last": float(price), "bid": None, "ask": None, "volume": None}
            except Exception:
                logger.exception("Tiingo get_quote failed, falling back to yfinance")
        # fallback: yfinance
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1d")
            if hist is None or hist.empty:
                return {"ticker": ticker, "time": None, "last": None, "bid": None, "ask": None, "volume": None}
            last_row = hist.iloc[-1]
            return {
                "ticker": ticker,
                "time": last_row.name.to_pydatetime(),
                "last": float(last_row['Close']),
                "bid": float(last_row.get('Open', last_row['Close'])),
                "ask": float(last_row.get('High', last_row['Close'])),
                "volume": int(last_row.get('Volume', 0))
            }
        except Exception:
            logger.exception("yfinance get_quote failed")
            return {"ticker": ticker, "time": None, "last": None, "bid": None, "ask": None, "volume": None}

    def get_history(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        if self._use_polygon:
            try:
                # polygon get historic aggregates; but to keep dependencies light we'll fallback
                pass
            except Exception:
                pass
        try:
            tk = yf.Ticker(ticker)
            return tk.history(period=period)
        except Exception:
            return pd.DataFrame()
