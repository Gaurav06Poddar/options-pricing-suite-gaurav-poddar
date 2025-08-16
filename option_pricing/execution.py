# option_pricing/execution.py
"""
Simple execution model:
 - execute_market(ticker, size, price=None, **kwargs) -> returns dict { 'ticker','size','price','tc',... }

Execution costs: tc_per_share + linear slippage per share + quadratic impact.
"""

from typing import Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SimpleExecutionEngine:
    def __init__(
        self,
        market_adapter,
        tc_per_share: float = 0.001,
        impact_per_share: float = 0.0,
        impact_coeff_quadratic: float = 0.0,
        liquidity_scale: float = 1e6,
    ):
        self.market = market_adapter
        self.tc_per_share = float(tc_per_share)
        self.impact_per_share = float(impact_per_share)
        self.impact_coeff_quadratic = float(impact_coeff_quadratic)
        self.liquidity_scale = float(liquidity_scale)

    def execute_market(self, ticker: str, size: float, price: float = None, **kwargs) -> Dict:
        """
        Execute a market order with optional metadata.
        
        Parameters
        ----------
        ticker : str
            Instrument identifier.
        size : float
            Quantity to buy (>0) or sell (<0).
        price : float, optional
            Override execution price (if None, use market mid).
        **kwargs : dict
            Extra metadata to store in the execution record (e.g. reason, strategy_name).
        
        Returns
        -------
        dict
            Execution record containing trade details and costs.
        """
        # Get current mid price if not provided
        q = self.market.get_quote(ticker)
        mid = float(q.get("last") or 0.0)
        exec_px = price if price is not None else mid

        # Transaction cost components
        tc_fixed = abs(size) * self.tc_per_share
        tc_linear = abs(size) * self.impact_per_share
        trade_value = abs(size) * exec_px
        tc_quad = self.impact_coeff_quadratic * (trade_value / self.liquidity_scale) ** 2
        total_tc = tc_fixed + tc_linear + tc_quad

        # Assume fully filled
        filled = size

        # Build execution record
        record = {
            "ticker": ticker,
            "size": float(filled),
            "price": float(exec_px),
            "tc": float(total_tc),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Merge in any metadata passed by strategies
        record.update(kwargs)

        logger.debug(f"Executed market order: {record}")
        return record
