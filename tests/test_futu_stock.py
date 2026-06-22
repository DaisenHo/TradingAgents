"""Tests for Futu OpenD-backed OHLCV and order book data."""

from __future__ import annotations

import copy
from unittest import mock

import pandas as pd
import pytest

import tradingagents.default_config as default_config
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def setup_function():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def _kline_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_key": ["2026-01-02 00:00:00", "2026-01-05 00:00:00"],
            "open": [179.771, 180.222],
            "high": [181.456, 182.111],
            "low": [178.8, 179.5],
            "close": [180.555, 181.666],
            "volume": [12300, 45600],
        }
    )


@pytest.mark.unit
class TestFutuProvider:
    def test_get_stock_data_fetches_daily_kline_from_opend(self, monkeypatch):
        from tradingagents.dataflows import futu_stock

        quote_ctx = mock.Mock()
        quote_ctx.request_history_kline.return_value = (futu_stock.RET_OK, _kline_frame(), None)
        quote_context_class = mock.Mock(return_value=quote_ctx)

        monkeypatch.setattr(futu_stock, "OpenQuoteContext", quote_context_class)

        out = futu_stock.get_stock_data("AAPL", "2026-01-01", "2026-01-10")

        quote_context_class.assert_called_once_with(host="127.0.0.1", port=11111)
        quote_ctx.request_history_kline.assert_called_once()
        args, kwargs = quote_ctx.request_history_kline.call_args
        assert args[0] == "US.AAPL"
        assert kwargs["start"] == "2026-01-01"
        assert kwargs["end"] == "2026-01-10"
        quote_ctx.close.assert_called_once()
        assert "# Stock data for US.AAPL from 2026-01-01 to 2026-01-10" in out
        assert "Date,Open,High,Low,Close,Volume" in out
        assert "2026-01-02,179.77,181.46,178.8,180.56,12300" in out

    def test_hk_symbol_is_converted_to_futu_market_code(self, monkeypatch):
        from tradingagents.dataflows import futu_stock

        quote_ctx = mock.Mock()
        quote_ctx.request_history_kline.return_value = (futu_stock.RET_OK, _kline_frame(), None)
        monkeypatch.setattr(futu_stock, "OpenQuoteContext", mock.Mock(return_value=quote_ctx))

        futu_stock.load_ohlcv("700.HK", "2026-01-01", "2026-01-10")

        args, _ = quote_ctx.request_history_kline.call_args
        assert args[0] == "HK.00700"

    def test_empty_futu_result_raises_no_market_data(self, monkeypatch):
        from tradingagents.dataflows import futu_stock

        quote_ctx = mock.Mock()
        quote_ctx.request_history_kline.return_value = (futu_stock.RET_OK, pd.DataFrame(), None)
        monkeypatch.setattr(futu_stock, "OpenQuoteContext", mock.Mock(return_value=quote_ctx))

        with pytest.raises(NoMarketDataError):
            futu_stock.load_ohlcv("AAPL", "2026-01-01", "2026-01-10")

    def test_get_order_book_subscribes_and_returns_snapshot(self, monkeypatch):
        from tradingagents.dataflows import futu_stock

        quote_ctx = mock.Mock()
        quote_ctx.subscribe.return_value = (futu_stock.RET_OK, "ok")
        quote_ctx.get_order_book.return_value = (
            futu_stock.RET_OK,
            {
                "code": "US.AAPL",
                "Bid": [(179.77, 100, 1, {})],
                "Ask": [(179.95, 400, 2, {})],
            },
        )
        monkeypatch.setattr(futu_stock, "OpenQuoteContext", mock.Mock(return_value=quote_ctx))

        out = futu_stock.get_order_book("AAPL")

        quote_ctx.subscribe.assert_called_once_with(["US.AAPL"], [futu_stock.SubType.ORDER_BOOK])
        assert "Bid" in out
        assert "179.77" in out
        quote_ctx.close.assert_called_once()


@pytest.mark.unit
class TestFutuRouting:
    def test_interface_exposes_futu_for_stock_data_and_indicators(self):
        from tradingagents.dataflows import interface

        assert "futu" in interface.VENDOR_METHODS["get_stock_data"]
        assert "futu" in interface.VENDOR_METHODS["get_indicators"]

    def test_stockstats_load_ohlcv_uses_futu_when_configured_first(self, tmp_path):
        from tradingagents.dataflows import stockstats_utils

        set_config(
            {
                "data_cache_dir": str(tmp_path),
                "data_vendors": {"technical_indicators": "futu,akshare,yfinance"},
            }
        )

        with mock.patch.object(
            stockstats_utils.futu_stock,
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
        ) as futu_loader, mock.patch.object(
            stockstats_utils.akshare_stock, "load_ohlcv"
        ) as ak_loader:
            data = stockstats_utils.load_ohlcv("AAPL", "2026-01-05")

        futu_loader.assert_called_once()
        ak_loader.assert_not_called()
        assert data["Close"].tolist() == [10.8, 11.1]
