# option_pricing/models_sql.py
"""
SQLAlchemy ORM models for portfolios, positions, trades, and performance snapshots.
Use Postgres backend (DATABASE_URL env var).
"""

import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

#DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@db:5432/options")
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    SQLITE_PATH = os.environ.get("LOCAL_DB_PATH", "data/options.db")
    os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"


engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    meta_json = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    positions = relationship("Position", back_populates="portfolio")
    trades = relationship("Trade", back_populates="portfolio")

class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    ticker = Column(String, index=True)
    qty = Column(Float, default=0.0)
    avg_price = Column(Float, default=0.0)
    last_update = Column(DateTime, default=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="positions")

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    ticker = Column(String, index=True)
    size = Column(Float)
    price = Column(Float)
    tc = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)
    meta_json = Column(JSON, default={})

    portfolio = relationship("Portfolio", back_populates="trades")

class PerfSnapshot(Base):
    __tablename__ = "perf_snapshots"
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    snapshot_time = Column(DateTime, default=datetime.utcnow)
    pnl = Column(Float)
    nav = Column(Float)
    metrics = Column(JSON, default={})

def init_db():
    Base.metadata.create_all(bind=engine)
