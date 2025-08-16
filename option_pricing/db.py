# option-pricing/db.py
import os
from sqlalchemy import (
    create_engine, meta_json, Table, Column,
    Integer, String, Float, DateTime, BigInteger, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError
import datetime
import json

DATABASE_URL = os.environ.get("DATABASE_URL")  # e.g. postgres://user:pass@host:5432/dbname

# Fallback to sqlite if DATABASE_URL not set
if DATABASE_URL is None or DATABASE_URL.strip() == "":
    SQLITE_PATH = os.environ.get("LOCAL_DB_PATH", "data/market.db")
    os.makedirs(os.path.dirname(SQLITE_PATH) or ".", exist_ok=True)
    DATABASE_URL = f"sqlite:///{SQLITE_PATH}"

# SQLAlchemy prefers "postgresql://" prefix
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# Existing historical price model for EOD / intraday OHLCV storage
class HistoricalPrice(Base):
    __tablename__ = "historical_prices"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ticker = Column(String(32), index=True, nullable=False)
    interval = Column(String(16), index=True, nullable=False)  # e.g., 1d,1m,5m
    timestamp = Column(DateTime, index=True, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

# Portfolio and Positions for persistence of user-defined portfolios
class PortfolioDef(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    # relationship to positions
    positions = relationship("PositionDef", back_populates="portfolio", cascade="all, delete-orphan")

class PositionDef(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False, index=True)
    kind = Column(String(32), nullable=False)  # 'underlying' or 'option'
    ticker = Column(String(32), nullable=True)
    qty = Column(Float, nullable=False, default=0.0)

    # option fields
    option_type = Column(String(16), nullable=True)  # call/put
    model = Column(String(32), nullable=True)  # 'bsm','binomial','mc'
    S = Column(Float, nullable=True)
    K = Column(Float, nullable=True)
    days = Column(Integer, nullable=True)
    r = Column(Float, nullable=True)
    sigma = Column(Float, nullable=True)
    extra = Column(Text, nullable=True)  # JSON-encoded extra dict

    portfolio = relationship("PortfolioDef", back_populates="positions")

def init_db():
    """
    Create tables if they don't exist.
    """
    Base.metadata.create_all(bind=engine)
