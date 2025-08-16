# option-pricing-models/strategies/short_straddle_delta_hedge.py
"""
Example plugin: short ATM straddle with delta-hedging.

Implements StrategyBase API.
"""

from option_pricing.strategy_base import StrategyBase
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ShortStraddleDeltaHedge(StrategyBase):
    meta_json = {"name": "ShortStraddleDeltaHedge", "author": "you", "version": "0.1"}
    default_config = {
        "ticker": "AAPL",
        "expiry": None,
        "notional": 10000.0,
        "rebalance_freq": 1,  # steps
        "sigma_model": 0.2,
    }

    def initialize(self, context):
        self.ctx = context
        self.ticker = self.config.get("ticker", "AAPL")
        self.rebalance_freq = int(self.config.get("rebalance_freq", 1))
        self.notional = float(self.config.get("notional", 10000.0))
        self.sigma_model = float(self.config.get("sigma_model", 0.2))
        # create initial positions (for demo, we don't actually buy real options here)
        self.ctx["logger"].info("ShortStraddle init: ticker=%s", self.ticker)

    def on_tick(self, context, market_data):
        # market_data is a dict with 'ticker' and 'last' price
        last = market_data.get("last")
        return  # no action on every tick for now

    def on_rebalance(self, context):
        # Ensure initialize has run
        if not hasattr(self, "ticker"):
            self.initialize(context)

        ticker = self.ticker
        spot = context["market"].get_quote(ticker)["last"]
        if spot is None:
            context["logger"].warning("No spot price available for %s, skipping rebalance", ticker)
            return

        K = round(spot)  # ATM strike
        days = 30
        r = 0.0
        sigma_model = self.sigma_model

        # compute deltas using package BlackScholesModel
        from option_pricing import BlackScholesModel

        call_bsm = BlackScholesModel(spot, K, days, r, sigma_model)
        call_delta = call_bsm.greeks("call")["delta"]
        put_delta = call_bsm.greeks("put")["delta"]
        net_delta = call_delta + put_delta  # hedge = -net_delta for shorting both

        hedge_shares = -net_delta * (self.notional / spot)

        # place market order via execution module
        context["execution"].execute_market(ticker, hedge_shares)
        context["logger"].info("Rebalanced: hedge_shares=%.4f for spot=%.2f", hedge_shares, spot)
