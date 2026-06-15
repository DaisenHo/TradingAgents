from __future__ import annotations

from datetime import datetime
from types import ModuleType
from typing import Annotated

import pandas as pd

from .symbol_utils import NoMarketDataError

_akshare: ModuleType | None = None


def _get_akshare() -> ModuleType:
    global _akshare
    if _akshare is None:
        try:
            import akshare
        except ImportError as exc:  # pragma: no cover - only without dependency
            raise ImportError(
                "akshare is not installed. Install the project dependencies or "
                "remove 'akshare' from the configured data vendor chain."
            ) from exc
        _akshare = akshare
    return _akshare


class _LazyAkshare:
    def __getattr__(self, name: str):
        return getattr(_get_akshare(), name)

    def __setattr__(self, name: str, value):
        setattr(_get_akshare(), name, value)

    def __delattr__(self, name: str):
        delattr(_get_akshare(), name)


ak = _LazyAkshare()


_COLUMN_MAP = {
    "日期": "Date",
    "date": "Date",
    "开盘": "Open",
    "open": "Open",
    "最高": "High",
    "high": "High",
    "最低": "Low",
    "low": "Low",
    "收盘": "Close",
    "close": "Close",
    "成交量": "Volume",
    "volume": "Volume",
}


def _compact_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%Y%m%d")


def _normalize_market_symbol(symbol: str) -> tuple[str, str, str]:
    raw = symbol.strip().upper()
    if not raw:
        raise ValueError("symbol must not be empty")

    for suffix in (".SS", ".SZ", ".BJ"):
        if raw.endswith(suffix):
            return "a_share", raw[: -len(suffix)], raw

    if raw.isdigit() and len(raw) == 6:
        return "a_share", raw, raw

    if raw.endswith(".HK"):
        code = raw[:-3].zfill(5)
        return "hk", code, raw

    return "us", raw, raw


def _a_share_exchange_symbol(symbol: str, label: str) -> str:
    if label.endswith(".SS"):
        return f"sh{symbol}"
    if label.endswith(".SZ"):
        return f"sz{symbol}"
    if label.endswith(".BJ"):
        return f"bj{symbol}"
    if symbol.startswith("6"):
        return f"sh{symbol}"
    if symbol.startswith(("4", "8")):
        return f"bj{symbol}"
    return f"sz{symbol}"


def _fetch_a_share_ohlcv(
    ak_symbol: str,
    label: str,
    start: str,
    end: str,
    adjust: str,
) -> pd.DataFrame:
    return _get_akshare().stock_zh_a_daily(
        symbol=_a_share_exchange_symbol(ak_symbol, label),
        start_date=start,
        end_date=end,
        adjust=adjust,
    )


def _fetch_us_ohlcv(
    ak_symbol: str,
    label: str,
    start: str,
    end: str,
    adjust: str,
) -> pd.DataFrame:
    return _get_akshare().stock_us_daily(symbol=label, adjust=adjust)


def _fetch_raw_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> tuple[pd.DataFrame, str, str]:
    market, ak_symbol, label = _normalize_market_symbol(symbol)
    start = _compact_date(start_date)
    end = _compact_date(end_date)

    if market == "a_share":
        data = _fetch_a_share_ohlcv(ak_symbol, label, start, end, adjust)
    elif market == "hk":
        data = _get_akshare().stock_hk_daily(symbol=ak_symbol, adjust=adjust)
    else:
        data = _fetch_us_ohlcv(ak_symbol, label, start, end, adjust)

    return data, label, ak_symbol


def _normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    normalized = data.rename(columns=_COLUMN_MAP)
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in normalized.columns]
    if missing:
        raise ValueError(f"AKShare result missing required columns: {missing}")

    normalized = normalized.loc[:, required].copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized = normalized.dropna(subset=["Date", "Close"])
    return normalized


def _filter_date_range(
    data: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    return data[(data["Date"] >= start) & (data["Date"] <= end)].copy()


def load_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    data, label, ak_symbol = _fetch_raw_ohlcv(symbol, start_date, end_date, adjust=adjust)
    normalized = _filter_date_range(_normalize_ohlcv(data), start_date, end_date)
    if normalized.empty:
        raise NoMarketDataError(symbol, ak_symbol, "AKShare returned no rows")
    return normalized


def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    data = load_ohlcv(symbol, start_date, end_date)
    rounded = data.copy()
    for col in ["Open", "High", "Low", "Close"]:
        rounded[col] = rounded[col].round(2)

    csv_string = rounded.to_csv(index=False)
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(rounded)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string
