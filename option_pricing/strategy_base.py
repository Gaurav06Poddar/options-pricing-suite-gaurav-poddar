#option-pricing-models/option_pricing/strategy_base.py
"""
Strategy plugin interface.

To implement a new strategy, create a subclass of StrategyBase and implement:
 - initialize(self, context)  # called once
 - on_tick(self, context, market_data)  # called on each market update
 - on_rebalance(self, context)           # called at scheduled rebalances
 - on_stop(self, context)                # called on shutdown

Context is a dict-like object containing:
 - 'portfolio_id', 'db', 'logger', 'market', 'execution', 'config', 'positions' ...
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class StrategyBase:
    meta_json = {"name": "BaseStrategy", "author": "you", "version": "0.1"}
    default_config = {}

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(self.default_config)
        self.config.update(config or {})
        self._alive = False

    def initialize(self, context: Dict[str, Any]):
        """One-time initialization. Create positions, subscribe symbols."""
        self.ctx = context

    def on_tick(self, context: Dict[str, Any], market_data: Dict[str, Any]):
        """Called on market quote; decide to place market/limit orders (use context['execution'])"""
        raise NotImplementedError()

    def on_rebalance(self, context: Dict[str, Any]):
        """Called at scheduled rebalance times (e.g., daily/minutely)"""
        raise NotImplementedError()

    def on_stop(self, context: Dict[str, Any]):
        """Cleanup before stopping"""
        pass

    def start(self, context: Dict[str, Any]):
        self._alive = True
        self.initialize(context)

    def stop(self):
        self._alive = False
