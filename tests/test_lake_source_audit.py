"""Lake source-coverage seal: classifier + exact predicates (fixtures only)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mascotrl.data.lake_source_audit import (
    FileClass,
    alternate_subset_ok,
    classify_csv_file,
    compare_year_counts,
    decide_verdict,
    header_fingerprint,
    list_downloads_csvs,
    primary_hist_cache_key,
    primary_hist_cache_valid,
    year_count_histogram_from_csv,
)


def test_classify_canonical_options(tmp_path: Path) -> None:
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    p = downloads / "option_prices_2003_2024_sp500_all.csv"
    p.write_text("date,secid\n2003-01-02,1\n", encoding="utf-8")
    tier_a = {"options_panel": p, "vol_surface": downloads / "missing.csv"}
    tier_b: dict[str, Path] = {}
    rec = classify_csv_file(p, downloads=downloads, tier_a=tier_a, tier_b=tier_b)
    assert rec["class"] == FileClass.CANONICAL_INGESTED
    assert rec["contract_key"] == "options_panel"


def test_classify_tier_b_duplicate_by_size_and_header(tmp_path: Path) -> None:
    downloads = tmp_path / "Downloads"
    data = tmp_path / "data"
    downloads.mkdir()
    data.mkdir()
    body = "PERMNO,date\n10001,2003-01-02\n"
    twin = data / "sp500_prices_2003_2024_all_constituents.csv"
    twin.write_text(body, encoding="utf-8")
    dump = downloads / "bnnmxdoysn5arpqe.csv"
    dump.write_text(body, encoding="utf-8")
    tier_b = {"sp500_prices": twin}
    rec = classify_csv_file(
        dump,
        downloads=downloads,
        tier_a={},
        tier_b=tier_b,
    )
    assert rec["class"] == FileClass.TIER_B_DUPLICATE
    assert rec["twin"] == str(twin)


def test_classify_alternate_om(tmp_path: Path) -> None:
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    p = downloads / "option_prices_slim_2003_2024.csv"
    p.write_text("date,secid\n", encoding="utf-8")
    rec = classify_csv_file(p, downloads=downloads, tier_a={}, tier_b={})
    assert rec["class"] == FileClass.ALTERNATE_OM_NOT_INGESTED


def test_compare_year_counts_exact_zero() -> None:
    left = {2003: 10, 2004: 20}
    right = {2003: 10, 2004: 20}
    ok, deltas = compare_year_counts(left, right, years=range(2003, 2005))
    assert ok is True
    assert deltas == {2003: 0, 2004: 0}


def test_compare_year_counts_fail_on_nonzero() -> None:
    ok, deltas = compare_year_counts(
        {2003: 10},
        {2003: 11},
        years=range(2003, 2004),
    )
    assert ok is False
    assert deltas[2003] == -1


def test_alternate_subset_ok() -> None:
    assert alternate_subset_ok({2003: 5, 2004: 0}, {2003: 10, 2004: 3}) is True
    assert alternate_subset_ok({2003: 11}, {2003: 10}) is False
    assert alternate_subset_ok({1999: 1}, {2003: 10}, years=range(2003, 2005)) is False


def test_primary_hist_cache_key_and_reuse(tmp_path: Path) -> None:
    src = tmp_path / "vsurd.csv"
    src.write_text("date\n2003-01-01\n", encoding="utf-8")
    key = primary_hist_cache_key(src)
    assert "size" in key and "mtime_ns" in key and "header_sha" in key
    cache = tmp_path / "hist.json"
    payload = {
        "cache_key": key,
        "rows_by_year": {"2003": 1},
        "n_rows": 1,
        "n_skip": 0,
    }
    cache.write_text(json.dumps(payload), encoding="utf-8")
    assert primary_hist_cache_valid(cache, src) is True
    src.write_text("date\n2003-01-01\n2004-01-01\n", encoding="utf-8")
    assert primary_hist_cache_valid(cache, src) is False


def test_year_histogram_count_only(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    p.write_text(
        "date,secid\n2003-01-02,1\n2003-06-01,2\n2004-01-03,3\nbad,4\n",
        encoding="utf-8",
    )
    hist = year_count_histogram_from_csv(p, date_col="date", year_start=2003, year_end=2024)
    assert hist["rows_by_year"] == {2003: 2, 2004: 1}
    assert hist["n_skip"] == 1
    assert hist["n_rows"] == 4


def test_decide_verdict_requires_wrds_for_complete() -> None:
    seals = {
        "S1": {"pass": True},
        "S2": {"pass": True},
        "S3": {"pass": True},
        "S4": {"pass": True},
        "S5": {"pass": False, "reason": "wrds_unverified"},
        "S6": {"pass": True},
    }
    verdict, code = decide_verdict(seals)
    assert verdict == "GAPS_DOCUMENTED"
    assert code != 0


def test_decide_verdict_complete_only_when_all_pass() -> None:
    seals = {f"S{i}": {"pass": True} for i in range(1, 7)}
    verdict, code = decide_verdict(seals)
    assert verdict == "COMPLETE"
    assert code == 0


def test_decide_verdict_wrds_newer_is_gaps() -> None:
    seals = {f"S{i}": {"pass": True} for i in range(1, 7)}
    seals["S5"] = {"pass": False, "reason": "WRDS_NEWER_THAN_LAKE"}
    verdict, code = decide_verdict(seals)
    assert verdict == "GAPS_DOCUMENTED"
    assert code != 0


def test_decide_verdict_fail_on_hard_seal() -> None:
    seals = {f"S{i}": {"pass": True} for i in range(1, 7)}
    seals["S2"] = {"pass": False, "reason": "year_mismatch"}
    verdict, code = decide_verdict(seals)
    assert verdict == "FAIL"
    assert code != 0


def test_classify_hash_mismatch_not_duplicate(tmp_path: Path) -> None:
    downloads = tmp_path / "Downloads"
    data = tmp_path / "data"
    downloads.mkdir()
    data.mkdir()
    twin = data / "sp500_prices_2003_2024_all_constituents.csv"
    twin.write_text("PERMNO,date\n10001,2003-01-02\n", encoding="utf-8")
    dump = downloads / "bnnmxdoysn5arpqe.csv"
    dump.write_text("PERMNO,date\n10001,2003-01-02\n10002,2003-01-03\n", encoding="utf-8")
    rec = classify_csv_file(
        dump,
        downloads=downloads,
        tier_a={},
        tier_b={"sp500_prices": twin},
    )
    assert rec["class"] != FileClass.TIER_B_DUPLICATE


def test_compare_year_counts_with_rejects_in_audit_module() -> None:
    from mascotrl.data.lake_source_audit import compare_year_counts_with_rejects

    ok, deltas = compare_year_counts_with_rejects(
        {2003: 100},
        {2003: 100},
        {},
        years=range(2003, 2004),
    )
    assert ok and deltas[2003] == 0


def test_header_fingerprint_stable(tmp_path: Path) -> None:
    p = tmp_path / "a.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    fp1 = header_fingerprint(p)
    fp2 = header_fingerprint(p)
    assert fp1 == fp2
    assert len(fp1) == 64


def test_s6_empty_listing_is_fail() -> None:
    from mascotrl.data.lake_source_audit import s6_listing_ok

    ok, reason = s6_listing_ok(files=[], unknown=[])
    assert ok is False
    assert reason == "empty_listing"
    ok, reason = s6_listing_ok(files=["a.csv"], unknown=[{"class": "UNKNOWN_MARKET"}])
    assert ok is False
    assert reason == "unknown_market"
    ok, reason = s6_listing_ok(files=["a.csv"], unknown=[])
    assert ok is True
    ok, reason = s6_listing_ok(
        files=["a.csv"], unknown=[], missing_contract=["vsurd_sp500_2003_2024.csv"]
    )
    assert ok is False
    assert reason == "missing_contract"


def test_list_downloads_csvs_flat_glob_not_nested_junk(tmp_path: Path) -> None:
    downloads = tmp_path / "Downloads"
    (downloads / "junk").mkdir(parents=True)
    (downloads / "option_prices_2003_2024_sp500_all.csv").write_text("date\n", encoding="utf-8")
    (downloads / "junk" / "sklearn.csv").write_text("a\n", encoding="utf-8")
    got = [p.name for p in list_downloads_csvs(downloads)]
    assert got == ["option_prices_2003_2024_sp500_all.csv"]


def test_classify_renamed_ibes_ratios_is_non_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate from the real lake (which may already hold the disclosure parquet).
    monkeypatch.setattr("mascotrl.data.lake_source_audit.LAKE_ROOT", tmp_path / "lake_empty")
    p = tmp_path / "non_contract" / "wrds_ibes_financial_ratios_sp500.csv"
    p.parent.mkdir()
    p.write_text("ticker,date,ratio\nAAPL,2003-01-02,1\n", encoding="utf-8")
    rec = classify_csv_file(p, downloads=tmp_path, tier_a={}, tier_b={})
    assert rec["class"] == FileClass.UNIQUE_NOT_INGESTED
    assert rec.get("non_contract") is True
    assert rec.get("disclosure_ingested") is not True


def test_classify_ibes_ratios_disclosure_ingested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lake = tmp_path / "lake"
    (lake / "macro").mkdir(parents=True)
    (lake / "macro" / "ibes_financial_ratios.parquet").write_bytes(b"PAR1")
    monkeypatch.setattr("mascotrl.data.lake_source_audit.LAKE_ROOT", lake)
    p = tmp_path / "non_contract" / "wrds_ibes_financial_ratios_sp500.csv"
    p.parent.mkdir()
    p.write_text("ticker,date,ratio\nAAPL,2003-01-02,1\n", encoding="utf-8")
    rec = classify_csv_file(p, downloads=tmp_path, tier_a={}, tier_b={})
    assert rec["class"] == FileClass.NON_MARKET
    assert rec.get("non_contract") is True
    assert rec.get("disclosure_ingested") is True


def test_list_downloads_csvs_rglob_under_volsurf_raw(tmp_path: Path) -> None:
    raw = tmp_path / "volsurf_raw"
    (raw / "om").mkdir(parents=True)
    (raw / "macro").mkdir(parents=True)
    (raw / "non_contract").mkdir(parents=True)
    (raw / "om" / "vsurd_sp500_2003_2024.csv").write_text("date\n", encoding="utf-8")
    (raw / "non_contract" / "wrds_ibes_financial_ratios_sp500.csv").write_text(
        "ticker,date\n", encoding="utf-8"
    )
    (raw / "macro" / "interest_rate_2003_2024_frb.csv").write_text("date\n", encoding="utf-8")
    names = {p.name for p in list_downloads_csvs(raw)}
    assert names == {
        "vsurd_sp500_2003_2024.csv",
        "wrds_ibes_financial_ratios_sp500.csv",
        "interest_rate_2003_2024_frb.csv",
    }
