# option-pricing/portfolio.py
"""
Portfolio utilities for options + underlyings.

Extended with DB persistence (SQLAlchemy) for saving/loading portfolio definitions.

See also: option-pricing/db.py which defines ORM models PortfolioDef and PositionDef.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json

# import your models
from .BlackScholesModel import BlackScholesModel
from .BinomialTreeModel import BinomialTreeModel
from .MonteCarloSimulation import MonteCarloPricing
from .ticker import Ticker

# DB session & models
from .db import SessionLocal, init_db, PortfolioDef, PositionDef

# ensure tables exist
init_db()

@dataclass
class Position:
    kind: str  # 'underlying' or 'option'
    ticker: Optional[str] = None   # optional for underlying, used for hints
    qty: float = 0.0               # positive = long, negative = short
    # option specific:
    option_type: Optional[str] = None  # 'call' or 'put'
    model: Optional[str] = None   # 'bsm', 'binomial', 'mc'
    S: Optional[float] = None
    K: Optional[float] = None
    days: Optional[int] = None
    r: Optional[float] = None
    sigma: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)  # e.g. mc sims, binomial steps

class Portfolio:
    def __init__(self):
        self.positions: List[Position] = []

    # -------------------------
    # Basic position manipulation
    # -------------------------
    def add_position(self, pos: Position):
        self.positions.append(pos)

    def remove_position(self, index: int):
        if 0 <= index < len(self.positions):
            self.positions.pop(index)

    def list_positions(self) -> pd.DataFrame:
        rows = []
        for i,p in enumerate(self.positions):
            if p.kind == 'underlying':
                rows.append({'idx': i, 'kind': 'underlying', 'ticker': p.ticker, 'qty': p.qty})
            else:
                rows.append({'idx': i, 'kind': 'option', 'type': p.option_type, 'model': p.model,
                             'S': p.S, 'K': p.K, 'days': p.days, 'r': p.r, 'sigma': p.sigma, 'qty': p.qty})
        return pd.DataFrame(rows)

    # -------------------------
    # Pricing & greeks (unchanged)
    # -------------------------
    def _price_and_greeks_for_pos(self, p: Position) -> Dict[str, float]:
        out = {'price': 0.0, 'delta': 0.0, 'gamma': None, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        if p.kind == 'underlying':
            S = float(p.S)
            out['price'] = S
            out['delta'] = 1.0
            out['gamma'] = 0.0
            out['vega'] = 0.0
            out['theta'] = 0.0
            out['rho'] = 0.0
            return out

        # option
        model_key = (p.model or 'bsm').lower()
        opt_type_value = 'Call Option' if (p.option_type and p.option_type.lower().startswith('c')) else 'Put Option'
        if model_key == 'bsm' or model_key == 'black-scholes' or model_key == 'black_scholes':
            bsm = BlackScholesModel(p.S, p.K, p.days, p.r, p.sigma)
            price_call = bsm.calculate_option_price('Call Option')
            price_put = bsm.calculate_option_price('Put Option')
            price = price_call if opt_type_value == 'Call Option' else price_put
            greeks = bsm.greeks('call') if opt_type_value == 'Call Option' else bsm.greeks('put')
            out.update({
                'price': price,
                'delta': greeks.get('delta', 0.0),
                'gamma': greeks.get('gamma', None),
                'vega': greeks.get('vega', 0.0),
                'theta': greeks.get('theta', 0.0),
                'rho': greeks.get('rho', 0.0),
            })
            return out

        if model_key == 'binomial':
            steps = int(p.extra.get('steps', 1000))
            american = bool(p.extra.get('american', False))
            B = BinomialTreeModel(p.S, p.K, p.days, p.r, p.sigma, number_of_time_steps=steps, american=american)
            price = B.calculate_option_price(opt_type_value)
            greeks = B.greeks('call') if opt_type_value == 'Call Option' else B.greeks('put')
            out.update({
                'price': price,
                'delta': greeks.get('delta', 0.0),
                'gamma': greeks.get('gamma', None),
                'vega': greeks.get('vega', 0.0),
                'theta': greeks.get('theta', 0.0),
                'rho': greeks.get('rho', 0.0),
            })
            return out

        if model_key in ('mc', 'montecarlo', 'monte-carlo'):
            sims = int(p.extra.get('sims', 20000))
            vr = p.extra.get('variance_reduction', 'none')
            MC = MonteCarloPricing(p.S, p.K, p.days, p.r, p.sigma, number_of_simulations=sims, variance_reduction=vr)
            MC.simulate_prices()
            price = MC.calculate_option_price('Call Option') if opt_type_value == 'Call Option' else MC.calculate_option_price('Put Option')
            greeks = MC.greeks(opt_type_value.lower().split()[0])
            out.update({
                'price': price,
                'delta': greeks.get('delta', 0.0),
                'gamma': greeks.get('gamma', None),
                'vega': greeks.get('vega', 0.0),
                'theta': greeks.get('theta', 0.0),
                'rho': greeks.get('rho', 0.0),
            })
            return out

        # fallback to BSM
        try:
            bsm = BlackScholesModel(p.S, p.K, p.days, p.r, p.sigma)
            price = bsm.calculate_option_price('Call Option') if opt_type_value == 'Call Option' else bsm.calculate_option_price('Put Option')
            greeks = bsm.greeks('call') if opt_type_value == 'Call Option' else bsm.greeks('put')
            out.update({
                'price': price,
                'delta': greeks.get('delta', 0.0),
                'gamma': greeks.get('gamma', None),
                'vega': greeks.get('vega', 0.0),
                'theta': greeks.get('theta', 0.0),
                'rho': greeks.get('rho', 0.0),
            })
            return out
        except Exception:
            return out

    def aggregated_metrics(self) -> Dict[str, float]:
        agg = {'price': 0.0, 'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
        gamma_missing = False
        for p in self.positions:
            metrics = self._price_and_greeks_for_pos(p)
            qty = float(p.qty)
            agg['price'] += qty * metrics.get('price', 0.0)
            agg['delta'] += qty * metrics.get('delta', 0.0)
            if metrics.get('gamma') is None:
                gamma_missing = True
            else:
                agg['gamma'] += qty * (metrics.get('gamma', 0.0) or 0.0)
            agg['vega'] += qty * (metrics.get('vega', 0.0) or 0.0)
            agg['theta'] += qty * (metrics.get('theta', 0.0) or 0.0)
            agg['rho'] += qty * (metrics.get('rho', 0.0) or 0.0)
        if gamma_missing:
            agg['gamma'] = None
        return agg

    def scenario_shock(self, shock_spot_pct=0.0, shock_vol_pct=0.0, shock_r=0.0) -> Dict[str, float]:
        base_metrics = self.aggregated_metrics()
        shocked_port = Portfolio()
        for p in self.positions:
            p2 = Position(**p.__dict__)
            if p2.S is not None:
                p2.S = float(p2.S) * (1.0 + shock_spot_pct)
            if p2.sigma is not None:
                p2.sigma = float(p2.sigma) + shock_vol_pct
            if p2.r is not None:
                p2.r = float(p2.r) + shock_r
            shocked_port.add_position(p2)
        shocked_metrics = shocked_port.aggregated_metrics()
        pnl = shocked_metrics['price'] - base_metrics['price']
        return {
            'base_price': base_metrics['price'],
            'shocked_price': shocked_metrics['price'],
            'pnL': pnl,
            'base_metrics': base_metrics,
            'shocked_metrics': shocked_metrics,
        }

    def export_positions_csv(self) -> str:
        df = self.list_positions()
        return df.to_csv(index=False)

    # -------------------------
    # Persistence: save/load portfolios to DB
    # -------------------------
    def save_to_db(self, name: str, overwrite: bool = True) -> None:
        """
        Save current portfolio into DB under 'name'. If overwrite=True, replace existing portfolio with same name.
        """
        session = SessionLocal()
        try:
            # find existing
            existing = session.query(PortfolioDef).filter(PortfolioDef.name == name).one_or_none()
            if existing is not None:
                if overwrite:
                    session.delete(existing)
                    session.flush()
                else:
                    raise ValueError(f"Portfolio with name '{name}' already exists. Use overwrite=True to replace.")
            # create new portfolio row
            pdrow = PortfolioDef(name=name)
            session.add(pdrow)
            session.flush()  # to get pdrow.id

            # add positions
            for p in self.positions:
                pos_extra = json.dumps(p.extra) if p.extra is not None else None
                posrow = PositionDef(
                    portfolio_id=pdrow.id,
                    kind=p.kind,
                    ticker=p.ticker,
                    qty=float(p.qty),
                    option_type=p.option_type,
                    model=p.model,
                    S=float(p.S) if p.S is not None else None,
                    K=float(p.K) if p.K is not None else None,
                    days=int(p.days) if p.days is not None else None,
                    r=float(p.r) if p.r is not None else None,
                    sigma=float(p.sigma) if p.sigma is not None else None,
                    extra=pos_extra
                )
                session.add(posrow)
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def list_saved_portfolios(self) -> List[str]:
        session = SessionLocal()
        try:
            rows = session.query(PortfolioDef).order_by(PortfolioDef.created_at.desc()).all()
            return [r.name for r in rows]
        finally:
            session.close()

    def load_from_db(self, name: str) -> None:
        """
        Load portfolio from DB into this Portfolio object (replaces current in-memory positions).
        """
        session = SessionLocal()
        try:
            pdrow = session.query(PortfolioDef).filter(PortfolioDef.name == name).one_or_none()
            if pdrow is None:
                raise ValueError(f"No portfolio named '{name}' found in DB.")
            # load positions
            positions = []
            for prow in pdrow.positions:
                extra = {}
                if prow.extra:
                    try:
                        extra = json.loads(prow.extra)
                    except Exception:
                        extra = {}
                p = Position(
                    kind=prow.kind,
                    ticker=prow.ticker,
                    qty=float(prow.qty),
                    option_type=prow.option_type,
                    model=prow.model,
                    S=float(prow.S) if prow.S is not None else None,
                    K=float(prow.K) if prow.K is not None else None,
                    days=int(prow.days) if prow.days is not None else None,
                    r=float(prow.r) if prow.r is not None else None,
                    sigma=float(prow.sigma) if prow.sigma is not None else None,
                    extra=extra
                )
                positions.append(p)
            # replace current positions
            self.positions = positions
        finally:
            session.close()

    def delete_portfolio(self, name: str) -> None:
        """
        Delete a saved portfolio from DB by name.
        """
        session = SessionLocal()
        try:
            pdrow = session.query(PortfolioDef).filter(PortfolioDef.name == name).one_or_none()
            if pdrow is None:
                raise ValueError(f"No portfolio named '{name}' found.")
            session.delete(pdrow)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
