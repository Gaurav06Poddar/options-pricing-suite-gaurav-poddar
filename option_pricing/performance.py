# option_pricing/performance.py
import numpy as np
import pandas as pd

def returns_from_nav(nav_series: pd.Series):
    return nav_series.pct_change().dropna()

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, freq: int = 252):
    r = returns - risk_free / freq
    mean = r.mean() * freq
    std = r.std(ddof=1) * np.sqrt(freq)
    if std == 0:
        return np.nan
    return float(mean / std)

def sortino_ratio(returns: pd.Series, mar: float = 0.0, freq: int = 252):
    r = returns - mar / freq
    mean = r.mean() * freq
    downside = r[r < 0].std(ddof=1) * np.sqrt(freq)
    if downside == 0:
        return np.nan
    return float(mean / downside)

def max_drawdown(nav: pd.Series):
    roll_max = nav.cummax()
    drawdown = (nav - roll_max) / roll_max
    return float(drawdown.min())

def turnover_from_trades(trades: pd.DataFrame, nav_series: pd.Series):
    """
    trades: DataFrame with ['timestamp','size','price']
    nav_series: series indexed by date/time
    returns turnover (annualized)
    """
    if trades.empty or nav_series.empty:
        return 0.0
    traded_notional = (trades['size'].abs() * trades['price']).sum()
    avg_nav = nav_series.mean()
    return float(traded_notional / avg_nav)

def performance_report(nav_series: pd.Series, trades: pd.DataFrame = None, risk_free=0.0):
    rets = returns_from_nav(nav_series)
    report = {
        'sharpe': sharpe_ratio(rets, risk_free),
        'sortino': sortino_ratio(rets, 0.0),
        'max_drawdown': max_drawdown(nav_series),
        'annualized_return': float((1.0 + rets.mean())**252 - 1.0) if not rets.empty else 0.0,
        'vol_annualized': float(rets.std(ddof=1) * np.sqrt(252)) if not rets.empty else 0.0,
    }
    if trades is not None:
        report['turnover'] = turnover_from_trades(trades, nav_series)
        report['total_notional_traded'] = float((trades['size'].abs() * trades['price']).sum())
        report['total_tc'] = float(trades['tc'].sum()) if 'tc' in trades.columns else 0.0
    return report
