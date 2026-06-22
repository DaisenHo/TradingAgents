from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Annotated, Any

import pandas as pd

from .errors import VendorNotConfiguredError
from .symbol_utils import NoMarketDataError

try:  # pragma: no cover - exercised when the optional SDK is installed
    from futu import RET_OK, AuType, KLType, OpenQuoteContext, SubType
except ImportError:  # pragma: no cover - unit tests monkeypatch the symbols
    RET_OK = 0

    class _MissingFutuConstant:
        K_DAY = "K_DAY"
        QFQ = "qfq"
        ORDER_BOOK = "ORDER_BOOK"

    AuType = _MissingFutuConstant
    KLType = _MissingFutuConstant
    OpenQuoteContext = None
    SubType = _MissingFutuConstant


_COLUMN_MAP = {
    "time_key": "Date",
    "date": "Date",
    "Date": "Date",
    "open": "Open",
    "Open": "Open",
    "high": "High",
    "High": "High",
    "low": "Low",
    "Low": "Low",
    "close": "Close",
    "Close": "Close",
    "volume": "Volume",
    "Volume": "Volume",
}


def _opend_host() -> str:
    return os.environ.get("FUTU_OPEND_HOST") or os.environ.get("FUTU_HOST") or "127.0.0.1"


def _opend_port() -> int:
    raw = os.environ.get("FUTU_OPEND_PORT") or os.environ.get("FUTU_PORT") or "11111"
    return int(raw)


def _quote_context():
    if OpenQuoteContext is None:
        raise VendorNotConfiguredError(
            "futu-api is not installed. Install futu-api and run Futu OpenD, "
            "or remove 'futu' from the configured data vendor chain."
        )
    return OpenQuoteContext(host=_opend_host(), port=_opend_port())


def normalize_futu_symbol(symbol: str) -> str:
    raw = symbol.strip().upper()
    if not raw:
        raise ValueError("symbol must not be empty")

    if raw.startswith(("US.", "HK.", "SH.", "SZ.", "BJ.")):
        market, code = raw.split(".", 1)
        if market == "HK" and code.isdigit():
            code = code.zfill(5)
        return f"{market}.{code}"

    if raw.endswith(".HK"):
        return f"HK.{raw[:-3].zfill(5)}"
    if raw.endswith(".SS"):
        return f"SH.{raw[:-3]}"
    if raw.endswith(".SZ"):
        return f"SZ.{raw[:-3]}"
    if raw.endswith(".BJ"):
        return f"BJ.{raw[:-3]}"
    if raw.endswith(".US"):
        return f"US.{raw[:-3]}"
    return f"US.{raw}"


def _normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    normalized = data.rename(columns=_COLUMN_MAP)
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in normalized.columns]
    if missing:
        raise ValueError(f"Futu result missing required columns: {missing}")

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


def _unpack_futu_response(response: tuple[Any, ...]) -> tuple[int, Any]:
    if len(response) < 2:
        raise RuntimeError(f"Unexpected Futu response shape: {response!r}")
    return response[0], response[1]


def load_ohlcv(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    futu_symbol = normalize_futu_symbol(symbol)
    quote_ctx = _quote_context()
    try:
        ret, data = _unpack_futu_response(
            quote_ctx.request_history_kline(
                futu_symbol,
                start=start_date,
                end=end_date,
                ktype=KLType.K_DAY,
                autype=AuType.QFQ,
            )
        )
    finally:
        quote_ctx.close()

    if ret != RET_OK:
        raise RuntimeError(f"Futu request_history_kline failed for {futu_symbol}: {data}")

    normalized = _filter_date_range(_normalize_ohlcv(data), start_date, end_date)
    if normalized.empty:
        raise NoMarketDataError(symbol, futu_symbol, "Futu returned no rows")
    return normalized


def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    futu_symbol = normalize_futu_symbol(symbol)
    data = load_ohlcv(symbol, start_date, end_date)
    rounded = data.copy()
    for col in ["Open", "High", "Low", "Close"]:
        rounded[col] = rounded[col].round(2)

    csv_string = rounded.to_csv(index=False)
    header = f"# Stock data for {futu_symbol} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(rounded)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    return header + csv_string


def get_order_book(
    symbol: Annotated[str, "ticker symbol of the company"],
) -> str:
    """Return a current order-book snapshot through Futu OpenD.

    Futu's real-time order-book API requires subscribing to ORDER_BOOK first;
    after subscription OpenD exposes the latest bid/ask snapshot.
    """
    futu_symbol = normalize_futu_symbol(symbol)
    quote_ctx = _quote_context()
    try:
        ret, data = _unpack_futu_response(
            quote_ctx.subscribe([futu_symbol], [SubType.ORDER_BOOK])
        )
        if ret != RET_OK:
            raise RuntimeError(f"Futu ORDER_BOOK subscribe failed for {futu_symbol}: {data}")

        ret, data = _unpack_futu_response(quote_ctx.get_order_book(futu_symbol))
        if ret != RET_OK:
            raise RuntimeError(f"Futu get_order_book failed for {futu_symbol}: {data}")
    finally:
        quote_ctx.close()

    return json.dumps(data, ensure_ascii=False, indent=2, default=str)
