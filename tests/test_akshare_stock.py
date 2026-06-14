"""Tests for AKShare-backed daily OHLCV data."""

from __future__ import annotations

import copy
import os
from unittest import mock

import pandas as pd
import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _ak_zh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "日期": ["2026-01-02", "2026-01-05"],
            "股票代码": ["000001", "000001"],
            "开盘": [10.111, 10.222],
            "收盘": [10.555, 10.666],
            "最高": [10.777, 10.888],
            "最低": [10.0, 10.1],
            "成交量": [12300, 45600],
        }
    )


def _ak_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05"],
            "open": [10.111, 10.222],
            "high": [10.777, 10.888],
            "low": [10.0, 10.1],
            "close": [10.555, 10.666],
            "volume": [12300, 45600],
        }
    )


def setup_function():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


@pytest.mark.unit
class TestAkshareProvider:
    def test_get_stock_data_uses_sina_for_a_share(self):
        from tradingagents.dataflows import akshare_stock

        with mock.patch.object(
            akshare_stock.ak,
            "stock_zh_a_hist",
            side_effect=AssertionError("eastmoney should not be called"),
        ) as hist:
            with mock.patch.object(
                akshare_stock.ak,
                "stock_zh_a_daily",
                return_value=_ak_daily_frame(),
            ) as sina:
                out = akshare_stock.get_stock_data(
                    "000001.SZ", "2026-01-01", "2026-01-10"
                )

        hist.assert_not_called()
        sina.assert_called_once_with(
            symbol="sz000001", start_date="20260101", end_date="20260110", adjust="qfq"
        )
        assert "# Stock data for 000001.SZ from 2026-01-01 to 2026-01-10" in out
        assert "Date,Open,High,Low,Close,Volume" in out
        assert "2026-01-02,10.11,10.78,10.0,10.56,12300" in out

    def test_bare_a_share_code_uses_sina_exchange_prefix(self):
        from tradingagents.dataflows import akshare_stock

        with mock.patch.object(
            akshare_stock.ak,
            "stock_zh_a_daily",
            return_value=_ak_daily_frame(),
        ) as sina:
            data = akshare_stock.load_ohlcv("600000", "2026-01-01", "2026-01-10")

        sina.assert_called_once_with(
            symbol="sh600000",
            start_date="20260101",
            end_date="20260110",
            adjust="qfq",
        )
        assert data["Close"].tolist() == [10.555, 10.666]

    def test_hk_suffix_is_converted_to_five_digit_akshare_symbol(self):
        from tradingagents.dataflows import akshare_stock

        with mock.patch.object(
            akshare_stock.ak,
            "stock_hk_hist",
            side_effect=AssertionError("eastmoney should not be called"),
        ) as hist:
            with mock.patch.object(
                akshare_stock.ak,
                "stock_hk_daily",
                return_value=_ak_daily_frame(),
            ) as sina:
                data = akshare_stock.load_ohlcv("700.HK", "2026-01-01", "2026-01-10")

        hist.assert_not_called()
        sina.assert_called_once_with(symbol="00700", adjust="qfq")
        assert list(data.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert pd.api.types.is_datetime64_any_dtype(data["Date"])

    def test_us_uses_sina_daily_and_filters_requested_dates(self):
        from tradingagents.dataflows import akshare_stock

        daily = pd.DataFrame(
            {
                "date": ["2025-12-31", "2026-01-02", "2026-01-05", "2026-01-11"],
                "open": [9.0, 10.111, 10.222, 11.0],
                "high": [9.5, 10.777, 10.888, 11.5],
                "low": [8.5, 10.0, 10.1, 10.5],
                "close": [9.2, 10.555, 10.666, 11.2],
                "volume": [900, 12300, 45600, 1100],
            }
        )

        with mock.patch.object(
            akshare_stock.ak,
            "stock_us_hist",
            side_effect=AssertionError("eastmoney should not be called"),
        ) as eastmoney:
            with mock.patch.object(
                akshare_stock.ak,
                "stock_us_spot_em",
                side_effect=AssertionError("eastmoney should not be called"),
            ) as spot:
                with mock.patch.object(
                    akshare_stock.ak,
                    "stock_us_daily",
                    return_value=daily,
                ) as sina:
                    data = akshare_stock.load_ohlcv("UNH", "2026-01-01", "2026-01-10")

        eastmoney.assert_not_called()
        spot.assert_not_called()
        sina.assert_called_once_with(symbol="UNH", adjust="qfq")
        assert data["Date"].dt.strftime("%Y-%m-%d").tolist() == [
            "2026-01-02",
            "2026-01-05",
        ]
        assert data["Close"].tolist() == [10.555, 10.666]

    def test_empty_akshare_result_raises_no_market_data(self):
        from tradingagents.dataflows import akshare_stock

        with mock.patch.object(
            akshare_stock.ak,
            "stock_zh_a_daily",
            return_value=pd.DataFrame(),
        ):
            with pytest.raises(NoMarketDataError):
                akshare_stock.load_ohlcv("000001.SZ", "2026-01-01", "2026-01-10")


@pytest.mark.unit
class TestAkshareRouting:
    def test_route_to_vendor_falls_back_from_akshare_to_yfinance(self):
        from tradingagents.dataflows import interface

        calls = []

        def raises_no_data(symbol, *args, **kwargs):
            calls.append(("akshare", symbol))
            raise NoMarketDataError(symbol, symbol, "no rows")

        def returns_yfinance(symbol, *args, **kwargs):
            calls.append(("yfinance", symbol))
            return "yf data"

        patched = {
            "akshare": raises_no_data,
            "yfinance": returns_yfinance,
        }
        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": patched},
            clear=False,
        ), mock.patch.object(interface, "get_vendor", return_value="akshare,yfinance"):
            result = interface.route_to_vendor(
                "get_stock_data", "000001.SZ", "2026-01-01", "2026-01-10"
            )

        assert result == "yf data"
        assert calls == [("akshare", "000001.SZ"), ("yfinance", "000001.SZ")]


@pytest.mark.unit
class TestLoadOhlcvAkshare:
    def test_load_ohlcv_uses_akshare_when_configured_first(self, tmp_path):
        from tradingagents.dataflows import stockstats_utils

        set_config(
            {
                "data_cache_dir": str(tmp_path),
                "data_vendors": {"technical_indicators": "akshare,yfinance"},
            }
        )

        with mock.patch.object(
            stockstats_utils.akshare_stock,
            "load_ohlcv",
            return_value=pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
                    "Open": [10.0, 10.5],
                    "High": [11.0, 11.5],
                    "Low": [9.5, 10.0],
                    "Close": [10.8, 11.1],
                    "Volume": [1000, 2000],
                }
            ),
        ) as ak_loader:
            with mock.patch.object(stockstats_utils.yf, "download") as yf_download:
                data = stockstats_utils.load_ohlcv("000001.SZ", "2026-01-05")

        ak_loader.assert_called_once()
        yf_download.assert_not_called()
        assert data["Close"].tolist() == [10.8, 11.1]
