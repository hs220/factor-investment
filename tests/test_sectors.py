"""Tests for sector classification: SEC SIC mapping + yfinance fallback."""
from __future__ import annotations

import pandas as pd

from src.data import fundamentals as F


def test_sic_to_sector_ranges():
    assert F._sic_to_sector("3571") == "Information Technology"  # computers
    assert F._sic_to_sector(6022) == "Financials"      # commercial banks
    assert F._sic_to_sector("4911") == "Utilities"
    assert F._sic_to_sector("5311") == "Consumer Discretionary"


def test_sic_to_sector_junk_is_none():
    for junk in (None, "", "NaN", "0000", "abc"):
        assert F._sic_to_sector(junk) is None


def test_yf_sector_mapping(monkeypatch):
    """yfinance vocab maps to our GICS labels; unknowns / errors -> None."""
    class _T:
        def __init__(self, ticker):
            self._t = ticker
        @property
        def info(self):
            return {
                "OZK": {"sector": "Financial Services"},
                "PBF": {"sector": "Energy"},
                "WAT": {"sector": "Mystery Sector"},   # unmapped vocab
            }[self._t]

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", _T)
    assert F._yf_sector("OZK") == "Financials"
    assert F._yf_sector("PBF") == "Energy"
    assert F._yf_sector("WAT") is None


def test_yf_sector_swallows_errors(monkeypatch):
    import yfinance

    def _boom(ticker):
        raise RuntimeError("network down")

    monkeypatch.setattr(yfinance, "Ticker", _boom)
    assert F._yf_sector("ANY") is None


def test_fetch_sectors_precedence(monkeypatch):
    """SEC SIC wins; blank SIC falls back to yfinance; both blank -> None."""
    monkeypatch.setattr(F, "_session", lambda: object())
    monkeypatch.setattr(F, "get_cik_map", lambda s: {"AAPL": "1", "OZK": "2", "ZZZ": "3"})
    subs = {
        "1": {"sic": "3571", "sicDescription": "Electronic Computers"},
        "2": {"sic": "", "sicDescription": ""},          # bank, blank SIC
        "3": {"sic": "", "sicDescription": ""},          # nothing anywhere
    }
    monkeypatch.setattr(F, "_fetch_submissions", lambda cik, s: subs[cik])
    monkeypatch.setattr(F, "_yf_sector", lambda t: {"OZK": "Financials"}.get(t))

    out = F.fetch_sectors(["AAPL", "OZK", "ZZZ"]).set_index("ticker")
    assert out.loc["AAPL", "sector_source"] == "sec_sic"
    assert out.loc["AAPL", "gics_sector"] in ("Industrials", "Information Technology")
    assert out.loc["OZK", "gics_sector"] == "Financials"
    assert out.loc["OZK", "sector_source"] == "yfinance"
    assert pd.isna(out.loc["ZZZ", "gics_sector"])
    assert pd.isna(out.loc["ZZZ", "sector_source"])


def test_yf_sector_map_targets_are_canonical():
    """Every fallback target is one of the 11 SIC-derived GICS labels."""
    sic_labels = {sec for _, _, sec in F._SIC_RANGES}
    assert set(F._YF_SECTOR_MAP.values()) <= sic_labels
