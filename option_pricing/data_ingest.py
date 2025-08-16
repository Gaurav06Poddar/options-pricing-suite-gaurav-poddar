# option-pricing/data_ingest.py
import os
from typing import Optional
import pandas as pd
import yfinance as yf
from sqlalchemy import select, delete
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

from .db import HistoricalPrice, SessionLocal, init_db

# initialize DB tables (no-op if exists)
init_db()

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure tz-naive and index named 'timestamp'
    if df.index.tz is not None:
        df = df.tz_convert(None)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'timestamp'
    return df

def fetch_and_store(ticker: str,
                    period: str = "1y",
                    interval: str = "1d",
                    overwrite: bool = False) -> pd.DataFrame:
    """
    Fetch from yfinance and store to DB (Postgres if configured, otherwise sqlite).
    If overwrite=True, existing rows for ticker+interval are deleted first.
    Returns the dataframe fetched.
    """
    tk = yf.Ticker(ticker)
    df = tk.history(period=period, interval=interval, prepost=False, actions=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} period={period} interval={interval}")

    df = _normalize_index(df)

    # Select columns we expect
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            df[col] = None

    # Convert to list of ORM objects
    records = []
    for ts, row in df.iterrows():
        rec = HistoricalPrice(
            ticker=ticker.upper(),
            interval=interval,
            timestamp=ts.to_pydatetime(),
            open=float(row['Open']) if pd.notna(row['Open']) else None,
            high=float(row['High']) if pd.notna(row['High']) else None,
            low=float(row['Low']) if pd.notna(row['Low']) else None,
            close=float(row['Close']) if pd.notna(row['Close']) else None,
            volume=float(row['Volume']) if pd.notna(row['Volume']) else None,
        )
        records.append(rec)

    session = SessionLocal()
    try:
        if overwrite:
            session.execute(delete(HistoricalPrice).where(HistoricalPrice.ticker == ticker.upper(), HistoricalPrice.interval == interval))
            session.commit()
        # Bulk insert
        # Use session.bulk_save_objects for efficiency
        session.bulk_save_objects(records)
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        raise
    finally:
        session.close()

    return df

def get_stored_history(ticker: str,
                       interval: str = "1d",
                       start: Optional[str] = None,
                       end: Optional[str] = None) -> pd.DataFrame:
    """
    Read stored history from DB into a DataFrame with DatetimeIndex.
    start/end are ISO strings or None.
    """
    session = SessionLocal()
    try:
        stmt = select(HistoricalPrice).where(HistoricalPrice.ticker == ticker.upper(), HistoricalPrice.interval == interval)
        if start:
            stmt = stmt.where(HistoricalPrice.timestamp >= pd.to_datetime(start).to_pydatetime())
        if end:
            stmt = stmt.where(HistoricalPrice.timestamp <= pd.to_datetime(end).to_pydatetime())
        stmt = stmt.order_by(HistoricalPrice.timestamp)
        rows = session.execute(stmt).scalars().all()
        if not rows:
            return pd.DataFrame()
        data = []
        for r in rows:
            data.append({
                'timestamp': r.timestamp,
                'Open': r.open,
                'High': r.high,
                'Low': r.low,
                'Close': r.close,
                'Volume': r.volume
            })
        df = pd.DataFrame(data).set_index(pd.to_datetime(pd.Series([d['timestamp'] for d in data])))
        df.index.name = 'timestamp'
        # Ensure column names consistent
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    finally:
        session.close()
