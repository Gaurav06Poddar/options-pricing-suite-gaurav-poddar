#option-pricing-models/option_pricing/strategy_runner.py
"""
Simple threaded strategy runner for paper trading.

- Manage multiple strategies
- Periodic tick (poll market via live_data.stream_quotes or scheduler)
- Periodic rebalance triggers
- Persist trades & positions via SQLAlchemy models (optional)
"""

import threading
import time
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class Runner:
    def __init__(self, market_adapter, execution_engine, db_session_factory=None):
        self.market = market_adapter
        self.execution = execution_engine
        self.db_factory = db_session_factory
        self.strategies = {}  # name -> instance
        self._threads = {}
        self._stop = threading.Event()

    def register(self, name: str, strategy):
        self.strategies[name] = strategy

    def start_all(self, tickers: List[str], interval_sec: int = 5, rebalance_freq_sec: int = 60):
        self._stop.clear()
        # Start polling thread
        self._threads['poller'] = threading.Thread(target=self._poll_loop, args=(tickers, interval_sec, rebalance_freq_sec), daemon=True)
        self._threads['poller'].start()
        logger.info("Runner started")

    def stop_all(self):
        self._stop.set()
        for t in self._threads.values():
            if t.is_alive():
                t.join(timeout=1.0)
        logger.info("Runner stopped")

    def _poll_loop(self, tickers, interval_sec, rebalance_freq_sec):
        last_rebalance = time.time()
        while not self._stop.is_set():
            # fetch quotes
            quotes = {}
            for t in tickers:
                quotes[t] = self.market.get_quote(t)
            # call each strategy on_tick
            for name, strat in self.strategies.items():
                try:
                    strat.on_tick({'market': self.market, 'execution': self.execution, 'logger': logger}, quotes)
                except Exception:
                    logger.exception("strategy on_tick failed")
            # rebalance scheduler
            if time.time() - last_rebalance >= rebalance_freq_sec:
                last_rebalance = time.time()
                for name, strat in self.strategies.items():
                    try:
                        strat.on_rebalance({'market': self.market, 'execution': self.execution, 'logger': logger})
                    except Exception:
                        logger.exception("strategy on_rebalance failed")
            time.sleep(interval_sec)
