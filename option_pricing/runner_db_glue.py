#option-pricing-models/option_pricing/runner_db_glue.py
"""
Glue layer that connects Runner -> Execution -> SQLAlchemy models for persistence.

Provides RunnerWithPersistence which wraps Runner and logs trades and perf snapshots.
"""

import logging
from typing import Dict, Any
from datetime import datetime
from option_pricing.strategy_runner import Runner
from option_pricing.models_sql import SessionLocal, Trade, PerfSnapshot, Position, Portfolio, init_db
import pandas as pd

logger = logging.getLogger(__name__)

class RunnerWithPersistence(Runner):
    def __init__(self, market_adapter, execution_engine, db_session_factory=SessionLocal, portfolio_name="demo_portfolio"):
        super().__init__(market_adapter, execution_engine, db_session_factory)
        self.db_session_factory = db_session_factory
        # ensure DB exists
        try:
            init_db()
        except Exception:
            logger.exception("init_db failed (maybe DB not configured)")
        # ensure portfolio record
        self.portfolio = None
        self.portfolio_name = portfolio_name
        self._ensure_portfolio()

    def _ensure_portfolio(self):
        try:
            db = self.db_session_factory()
            p = db.query(Portfolio).filter(Portfolio.name == self.portfolio_name).first()
            if p is None:
                p = Portfolio(name=self.portfolio_name, meta_json={})
                db.add(p); db.commit(); db.refresh(p)
            self.portfolio = p
            db.close()
        except Exception:
            logger.exception("Could not ensure portfolio table")

    def persist_trade(self, ticker: str, size: float, price: float, tc: float = 0.0, meta_json: Dict[str, Any] = None):
        meta_json = meta_json or {}
        try:
            db = self.db_session_factory()
            tr = Trade(portfolio_id=self.portfolio.id, ticker=ticker, size=size, price=price, tc=tc, meta_json=meta_json)
            db.add(tr); db.commit()
            db.close()
        except Exception:
            logger.exception("persist_trade failed")

    def persist_snapshot(self, nav: float, pnl: float, metrics: Dict[str, Any]):
        try:
            db = self.db_session_factory()
            snap = PerfSnapshot(portfolio_id=self.portfolio.id, snapshot_time=datetime.utcnow(), pnl=pnl, nav=nav, metrics=metrics)
            db.add(snap); db.commit()
            db.close()
        except Exception:
            logger.exception("persist_snapshot failed")

    # override Runner behavior to intercept executions
    def start_all(self, tickers, interval_sec: int = 5, rebalance_freq_sec: int = 60):
        # wrap execution to persist trades
        original_execute = getattr(self.execution, "execute_market", None)
        if original_execute is not None:
            def wrapped_execute(ticker, size, price=None):
                rec = original_execute(ticker, size, price)
                # save to DB
                try:
                    self.persist_trade(rec['ticker'], rec['size'], rec['price'], rec.get('tc', 0.0), meta_json={})
                except Exception:
                    logger.exception("persist_trade in wrapped_execute failed")
                return rec
            # monkeypatch
            self.execution.execute_market = wrapped_execute
        super().start_all(tickers, interval_sec=interval_sec, rebalance_freq_sec=rebalance_freq_sec)

    # convenience: expose recent trades DataFrame
    def trades_df(self, limit=1000):
        try:
            db = self.db_session_factory()
            rows = db.query(Trade).filter(Trade.portfolio_id == self.portfolio.id).order_by(Trade.timestamp.desc()).limit(limit).all()
            db.close()
            if not rows:
                return pd.DataFrame()
            data = []
            for r in rows:
                data.append({'id': r.id, 'ticker': r.ticker, 'size': r.size, 'price': r.price, 'tc': r.tc, 'timestamp': r.timestamp})
            return pd.DataFrame(data)
        except Exception:
            logger.exception("trades_df failed")
            return pd.DataFrame()
