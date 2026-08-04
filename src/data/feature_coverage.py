"""Column-coverage audit: every lake column maps to a consumer or an unused reason."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.data.paths import LAKE_ROOT, assert_lake_mounted

# Flat parquet tables under lake/ that the audit walks.
FLAT_TABLES: tuple[str, ...] = (
    "factors/jkp_chars.parquet",
    "macro/cboe_vix.parquet",
    "macro/compustat_funda_enrich.parquet",
    "macro/crsp_om_adv.parquet",
    "macro/crsp_optionm_link.parquet",
    "macro/ff_factors.parquet",
    "macro/ibes_financial_ratios.parquet",
    "macro/interest_rate.parquet",
    "macro/lseg_eq_ohlc_corax.parquet",
    "macro/lseg_eq_ohlc_unadj.parquet",
    "macro/lseg_eq_size.parquet",
    "macro/lseg_gics.parquet",
    "macro/lseg_index_vol_rates.parquet",
    "macro/lseg_ric_map.parquet",
    "macro/lseg_spx_pit.parquet",
    "macro/om_borrate.parquet",
    "macro/om_distrd.parquet",
    "macro/om_idxdvd.parquet",
    "macro/om_option_adv.parquet",
    "macro/om_opvold.parquet",
    "macro/om_secnmd.parquet",
    "macro/om_securd.parquet",
    "macro/om_stdbrte.parquet",
    "macro/om_stdopd.parquet",
    "macro/om_zerocd.parquet",
    "macro/p3/lseg_ibes.parquet",
    "macro/p3/lseg_short_interest.parquet",
    "macro/p3/lseg_worldscope.parquet",
    "macro/pastor_stambaugh.parquet",
    "macro/pit_membership.parquet",
    "macro/sp500_fwd.parquet",
    "macro/sp500_hv.parquet",
    "macro/sp500_prices.parquet",
    "macro/sp500_sec.parquet",
    "macro/spx_index.parquet",
    "macro/spx_opvold.parquet",
    "macro/spx_zerocd.parquet",
)

HIVE_TABLES: tuple[str, ...] = ("options_panel", "vol_surface")

# Column key = "relative/path.parquet::col" or "HIVE:name::col"
CONSUMER_MAP: dict[str, str] = {
    # --- equity spine / returns ---
    "macro/sp500_sec.parquet::secid": "equity_panel / feature_panels joins",
    "macro/sp500_sec.parquet::date": "equity_panel / feature_panels joins",
    "macro/sp500_sec.parquet::close": "equity_panel FEATURE_STEMS; feature_panels price joins",
    "macro/sp500_sec.parquet::volume": "equity_panel dollar_volume; feature_panels.option_flow stock_vol",
    "macro/sp500_sec.parquet::return": "equity_panel stk_ret; assemble returns",
    "macro/sp500_sec.parquet::cfadj": "equity_panel cfadj validation; Yang-Zhang adj close",
    "macro/sp500_sec.parquet::shrout": "equity_panel mktcap",
    "macro/sp500_sec.parquet::ticker": "equity_panel PIT membership join",
    "macro/sp500_sec.parquet::open": "feature_panels.ohlc fallback",
    "macro/sp500_sec.parquet::high": "feature_panels.ohlc fallback",
    "macro/sp500_sec.parquet::low": "feature_panels.ohlc fallback",
    "macro/sp500_sec.parquet::lseg_bid": "sp500_sec LSEG overlay (microstructure parallel)",
    "macro/sp500_sec.parquet::lseg_ask": "sp500_sec LSEG overlay (microstructure parallel)",
    "macro/sp500_sec.parquet::lseg_trdprc": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_open": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_high": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_low": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_trnvr": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_num_moves": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_acvol": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_vwap": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_vwap_vol": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_blkcount": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_blkvolum": "sp500_sec LSEG overlay",
    "macro/sp500_sec.parquet::lseg_quoted_spread": "sp500_sec LSEG overlay spread",
    # --- LSEG OHLC ---
    "macro/lseg_eq_ohlc_unadj.parquet::date": "feature_panels.ohlc / microstructure",
    "macro/lseg_eq_ohlc_unadj.parquet::secid": "feature_panels.ohlc / microstructure",
    "macro/lseg_eq_ohlc_unadj.parquet::OPEN_PRC": "feature_panels.ohlc / range_volatility",
    "macro/lseg_eq_ohlc_unadj.parquet::HIGH_1": "feature_panels.ohlc / range_volatility",
    "macro/lseg_eq_ohlc_unadj.parquet::LOW_1": "feature_panels.ohlc / range_volatility",
    "macro/lseg_eq_ohlc_unadj.parquet::TRDPRC_1": "feature_panels.ohlc / microstructure vwap_dev",
    "macro/lseg_eq_ohlc_unadj.parquet::BID": "feature_panels.microstructure eff_spread",
    "macro/lseg_eq_ohlc_unadj.parquet::ASK": "feature_panels.microstructure eff_spread",
    "macro/lseg_eq_ohlc_unadj.parquet::VWAP": "feature_panels.microstructure vwap_dev",
    "macro/lseg_eq_ohlc_unadj.parquet::ACVOL_UNS": "feature_panels.microstructure turnover/block",
    "macro/lseg_eq_ohlc_unadj.parquet::BLKVOLUM": "feature_panels.microstructure block_share",
    "macro/lseg_eq_ohlc_unadj.parquet::BLKCOUNT": "feature_panels.microstructure block_share",
    "macro/lseg_eq_ohlc_corax.parquet::date": "feature_panels.ohlc Yang-Zhang overnight prefer",
    "macro/lseg_eq_ohlc_corax.parquet::secid": "feature_panels.ohlc Yang-Zhang overnight prefer",
    "macro/lseg_eq_ohlc_corax.parquet::OPEN_PRC": "feature_panels.ohlc Yang-Zhang overnight prefer",
    "macro/lseg_eq_ohlc_corax.parquet::HIGH_1": "feature_panels.ohlc Yang-Zhang overnight prefer",
    "macro/lseg_eq_ohlc_corax.parquet::LOW_1": "feature_panels.ohlc Yang-Zhang overnight prefer",
    "macro/lseg_eq_ohlc_corax.parquet::TRDPRC_1": "feature_panels.ohlc adj close prefer",
    # --- size ---
    "macro/lseg_eq_size.parquet::date": "feature_panels.microstructure turnover",
    "macro/lseg_eq_size.parquet::secid": "feature_panels.microstructure turnover",
    "macro/lseg_eq_size.parquet::Outstanding Shares": "feature_panels.microstructure turnover",
    "macro/lseg_eq_size.parquet::Company Market Cap": "feature_panels size / jkp parallel",
    # --- GICS / short / analyst / worldscope ---
    "macro/lseg_gics.parquet::Instrument": "feature_panels.gics_map",
    "macro/lseg_gics.parquet::TR.GICSIndustry": "feature_panels.gics_map / experimental gics_rel_mom",
    "macro/lseg_gics.parquet::TR.GICSSector": "feature_panels.gics_map",
    "macro/lseg_gics.parquet::TR.GICSSubIndustry": "feature_panels.gics_map",
    "macro/p3/lseg_short_interest.parquet::date": "feature_panels.short_interest",
    "macro/p3/lseg_short_interest.parquet::secid": "feature_panels.short_interest",
    "macro/p3/lseg_short_interest.parquet::Short Interest Pct": "feature_panels.short_interest si_pct",
    "macro/p3/lseg_short_interest.parquet::Short Interest": "feature_panels.short_interest level",
    "macro/p3/lseg_ibes.parquet::date": "feature_panels.analyst",
    "macro/p3/lseg_ibes.parquet::secid": "feature_panels.analyst",
    "macro/p3/lseg_ibes.parquet::Price Target - Median": "feature_panels.analyst pt_gap",
    "macro/p3/lseg_ibes.parquet::Recommendation - Mean (1-5)": "feature_panels.analyst rec_mean",
    "macro/p3/lseg_worldscope.parquet::date": "feature_panels.worldscope",
    "macro/p3/lseg_worldscope.parquet::secid": "feature_panels.worldscope",
    "macro/p3/lseg_worldscope.parquet::Book Value Per Share": "feature_panels.worldscope bp",
    "macro/p3/lseg_worldscope.parquet::P/E (Daily Time Series Ratio)": "feature_panels.worldscope ep",
    "macro/p3/lseg_worldscope.parquet::Revenue": "feature_panels.worldscope rev_growth",
    "macro/p3/lseg_worldscope.parquet::Total Assets": "feature_panels.worldscope ta_growth",
    "macro/p3/lseg_worldscope.parquet::Total Equity": "feature_panels.worldscope bp alt",
    # --- IBES curated 12 ---
    "macro/ibes_financial_ratios.parquet::public_date": "feature_panels.ibes_ratios PIT",
    "macro/ibes_financial_ratios.parquet::permno": "feature_panels.ibes_ratios link",
    "macro/ibes_financial_ratios.parquet::gvkey": "feature_panels.ibes_ratios link",
    "macro/ibes_financial_ratios.parquet::bm": "feature_panels.ibes_ratios",
    "macro/ibes_financial_ratios.parquet::pe_exi": "feature_panels.ibes_ratios ep_exi",
    "macro/ibes_financial_ratios.parquet::ps": "feature_panels.ibes_ratios",
    "macro/ibes_financial_ratios.parquet::pcf": "feature_panels.ibes_ratios",
    "macro/ibes_financial_ratios.parquet::dpr": "feature_panels.ibes_ratios",
    "macro/ibes_financial_ratios.parquet::npm": "feature_panels.ibes_ratios",
    "macro/ibes_financial_ratios.parquet::gpm": "feature_panels.ibes_ratios",
    "macro/ibes_financial_ratios.parquet::roa": "feature_panels.ibes_ratios",
    "macro/ibes_financial_ratios.parquet::roe": "feature_panels.ibes_ratios",
    "macro/ibes_financial_ratios.parquet::cfm": "feature_panels.ibes_ratios",
    "macro/ibes_financial_ratios.parquet::evm": "feature_panels.ibes_ratios",
    "macro/ibes_financial_ratios.parquet::CAPEI": "feature_panels.ibes_ratios capex_inv",
    # --- Compustat ---
    "macro/compustat_funda_enrich.parquet::datadate": "feature_panels.compustat PIT lag",
    "macro/compustat_funda_enrich.parquet::secid": "feature_panels.compustat",
    "macro/compustat_funda_enrich.parquet::permno": "feature_panels.compustat link",
    "macro/compustat_funda_enrich.parquet::at": "feature_panels.compustat at_growth",
    "macro/compustat_funda_enrich.parquet::sale": "feature_panels.compustat sale_growth",
    "macro/compustat_funda_enrich.parquet::ni": "feature_panels.compustat ni_at",
    "macro/compustat_funda_enrich.parquet::dvc": "feature_panels.compustat dvc_at",
    # --- option flow / dividend / rates ---
    "macro/om_opvold.parquet::secid": "feature_panels.option_flow",
    "macro/om_opvold.parquet::date": "feature_panels.option_flow",
    "macro/om_opvold.parquet::cp_flag": "feature_panels.option_flow",
    "macro/om_opvold.parquet::volume": "feature_panels.option_flow",
    "macro/om_opvold.parquet::open_interest": "feature_panels.option_flow",
    "macro/om_distrd.parquet::secid": "feature_panels.dividend",
    "macro/om_distrd.parquet::ex_date": "feature_panels.dividend",
    "macro/om_distrd.parquet::amount": "feature_panels.dividend",
    "macro/om_zerocd.parquet::date": "feature_panels.rates_term",
    "macro/om_zerocd.parquet::days": "feature_panels.rates_term",
    "macro/om_zerocd.parquet::rate": "feature_panels.rates_term",
    "macro/spx_zerocd.parquet::date": "feature_panels.rates_term fallback",
    "macro/spx_zerocd.parquet::days": "feature_panels.rates_term fallback",
    "macro/spx_zerocd.parquet::rate": "feature_panels.rates_term fallback",
    # --- JKP / FF / ADV / borrow / surface ---
    "factors/jkp_chars.parquet::date": "feature_panels.jkp",
    "factors/jkp_chars.parquet::permno": "feature_panels.jkp link",
    "factors/jkp_chars.parquet::me": "feature_panels.jkp log_me",
    "factors/jkp_chars.parquet::ivol_capm_21d": "feature_panels.jkp",
    "factors/jkp_chars.parquet::ret_1_0": "feature_panels.jkp",
    "macro/ff_factors.parquet::date": "residualization / resid_mom / idio_vol",
    "macro/ff_factors.parquet::Mkt-RF": "residualization / beta_asym",
    "macro/ff_factors.parquet::SMB": "residualization",
    "macro/ff_factors.parquet::HML": "residualization",
    "macro/ff_factors.parquet::Mom": "residualization FF4",
    "macro/ff_factors.parquet::RF": "residualization",
    "macro/crsp_om_adv.parquet::date": "liquidity ADV / campaign dollar_volume",
    "macro/crsp_om_adv.parquet::secid": "liquidity ADV",
    "macro/crsp_om_adv.parquet::adv": "liquidity ADV",
    "macro/om_borrate.parquet::secid": "iv_surface borrow / assemble borrow",
    "macro/om_borrate.parquet::date": "iv_surface borrow",
    "macro/om_borrate.parquet::borrowrate": "iv_surface borrow",
    "macro/om_borrate.parquet::days": "iv_surface borrow tenor select",
    "macro/cboe_vix.parquet::Date": "macro_loader VIX",
    "macro/cboe_vix.parquet::vix": "macro_loader VIX",
    "macro/interest_rate.parquet::date": "macro_loader rates; lseg overlay",
    "macro/interest_rate.parquet::dtb3": "macro_loader rates",
    "macro/interest_rate.parquet::dgs10": "macro_loader rates",
    "macro/interest_rate.parquet::sofr": "macro_loader rates",
    "macro/interest_rate.parquet::lseg_us2y": "lseg overlay rates",
    "macro/interest_rate.parquet::lseg_us10y": "lseg overlay rates",
    "macro/interest_rate.parquet::lseg_sofr": "lseg overlay rates",
    "macro/pastor_stambaugh.parquet::DATE": "liquidity factor research",
    "macro/pastor_stambaugh.parquet::PS_LEVEL": "liquidity factor research",
    "macro/pastor_stambaugh.parquet::PS_INNOV": "liquidity factor research",
    "macro/pit_membership.parquet::ticker": "equity_panel PIT membership",
    "macro/pit_membership.parquet::start_date": "equity_panel PIT membership",
    "macro/pit_membership.parquet::end_date": "equity_panel PIT membership",
    "macro/sp500_hv.parquet::secid": "campaign HV aux / surface VRP",
    "macro/sp500_hv.parquet::date": "campaign HV aux",
    "macro/sp500_hv.parquet::days": "campaign HV tenor",
    "macro/sp500_hv.parquet::volatility": "campaign HV aux",
    "macro/crsp_optionm_link.parquet::secid": "WRDS link / jkp permno join",
    "macro/crsp_optionm_link.parquet::permno": "WRDS link / jkp permno join",
    "macro/crsp_optionm_link.parquet::sdate": "WRDS link window",
    "macro/crsp_optionm_link.parquet::edate": "WRDS link window",
    "macro/lseg_index_vol_rates.parquet::date": "lseg overlay interest_rate",
    "macro/lseg_index_vol_rates.parquet::ric": "lseg overlay interest_rate",
    "macro/lseg_index_vol_rates.parquet::YLDTOMAT": "lseg overlay interest_rate",
    "macro/lseg_index_vol_rates.parquet::FIXING_1": "lseg overlay interest_rate",
    "macro/lseg_index_vol_rates.parquet::asof_ts": "lseg overlay interest_rate provenance",
    "macro/lseg_ric_map.parquet::secid": "identifier join / GICS map",
    "macro/lseg_ric_map.parquet::ric": "identifier join / GICS map",
    "macro/lseg_ric_map.parquet::TR.GICSIndustry": "feature_panels.gics_map via ric",
    "macro/lseg_spx_pit.parquet::Constituent RIC": "LSEG PIT membership parallel",
    "macro/lseg_spx_pit.parquet::pit_date": "LSEG PIT membership parallel",
    "macro/om_option_adv.parquet::secid": "option universe ADV screen",
    "macro/om_option_adv.parquet::date": "option universe ADV screen",
    "macro/om_option_adv.parquet::opt_volume": "option universe ADV screen",
    "macro/om_option_adv.parquet::opt_open_interest": "option universe ADV screen",
    "macro/om_stdopd.parquet::secid": "surface / std option marks",
    "macro/om_stdopd.parquet::date": "surface / std option marks",
    "macro/om_stdopd.parquet::impl_volatility": "surface / std option marks",
    "macro/om_stdopd.parquet::delta": "surface / std option marks",
    "macro/om_stdopd.parquet::days": "surface / std option marks",
    "macro/om_stdopd.parquet::cp_flag": "surface / std option marks",
    "macro/om_stdbrte.parquet::secid": "borrow fallback",
    "macro/om_stdbrte.parquet::date": "borrow fallback",
    "macro/om_stdbrte.parquet::borrowrate": "borrow fallback",
    "macro/om_idxdvd.parquet::secid": "dividend index rate aux",
    "macro/om_idxdvd.parquet::date": "dividend index rate aux",
    "macro/om_idxdvd.parquet::rate": "dividend index rate aux",
    "macro/spx_opvold.parquet::secid": "index option flow aux",
    "macro/spx_opvold.parquet::date": "index option flow aux",
    "macro/spx_opvold.parquet::cp_flag": "index option flow aux",
    "macro/spx_opvold.parquet::volume": "index option flow aux",
    "macro/spx_opvold.parquet::open_interest": "index option flow aux",
    # --- hive ---
    "HIVE:options_panel::secid": "duckdb_engine marks / OOS panel",
    "HIVE:options_panel::date": "duckdb_engine marks / OOS panel",
    "HIVE:options_panel::cp_flag": "duckdb_engine marks",
    "HIVE:options_panel::strike_price": "duckdb_engine marks",
    "HIVE:options_panel::best_bid": "duckdb_engine marks / friction",
    "HIVE:options_panel::best_offer": "duckdb_engine marks / friction",
    "HIVE:options_panel::volume": "duckdb_engine / ADV",
    "HIVE:options_panel::open_interest": "duckdb_engine",
    "HIVE:options_panel::impl_volatility": "duckdb_engine marks",
    "HIVE:options_panel::delta": "duckdb_engine marks / DHGNN",
    "HIVE:options_panel::exdate": "duckdb_engine tenor",
    "HIVE:options_panel::optionid": "duckdb_engine identity",
    "HIVE:options_panel::forward_price": "duckdb_engine marks",
    "HIVE:vol_surface::secid": "surface_signals / Kelly images",
    "HIVE:vol_surface::date": "surface_signals / Kelly images",
    "HIVE:vol_surface::days": "surface_signals / Kelly tenors",
    "HIVE:vol_surface::delta": "surface_signals / Kelly deltas",
    "HIVE:vol_surface::impl_volatility": "surface_signals / Kelly images",
    "HIVE:vol_surface::cp_flag": "surface_signals",
    "HIVE:vol_surface::dispersion": "surface_signals surface_dispersion",
    "HIVE:vol_surface::impl_strike": "surface_signals geometry",
    "HIVE:vol_surface::impl_premium": "surface_signals geometry",
}

# Columns that must never claim a silent consumer.
INTENTIONALLY_UNUSED: dict[str, str] = {
    # identifiers / provenance
    "macro/sp500_sec.parquet::cusip": "identifier only",
    "macro/sp500_sec.parquet::sic": "industry code unused by FEAT",
    "macro/sp500_sec.parquet::index_flag": "OM metadata",
    "macro/sp500_sec.parquet::exchange_d": "OM metadata",
    "macro/sp500_sec.parquet::class": "OM metadata",
    "macro/sp500_sec.parquet::issue_type": "OM metadata",
    "macro/sp500_sec.parquet::industry_group": "OM metadata",
    "macro/sp500_sec.parquet::cfret": "OM metadata; return SoT is return col",
    "macro/sp500_sec.parquet::lseg_trd_status": "QA flag",
    "macro/sp500_sec.parquet::lseg_ric": "identifier",
    "macro/sp500_sec.parquet::lseg_asof_ts": "provenance",
    "macro/lseg_eq_ohlc_unadj.parquet::TRNOVR_UNS": "turnover dollar unused; share turnover from ACVOL",
    "macro/lseg_eq_ohlc_unadj.parquet::NUM_MOVES": "trade count unused by FEAT",
    "macro/lseg_eq_ohlc_unadj.parquet::VWAP_VOL": "VWAP volume unused; ACVOL used",
    "macro/lseg_eq_ohlc_unadj.parquet::TRD_STATUS": "QA flag",
    "macro/lseg_eq_ohlc_unadj.parquet::ric": "identifier",
    "macro/lseg_eq_ohlc_unadj.parquet::asof_ts": "provenance",
    "macro/lseg_eq_ohlc_corax.parquet::BID": "corax preferred for adj close/open; bid/ask from unadj",
    "macro/lseg_eq_ohlc_corax.parquet::ASK": "corax preferred for adj close/open; bid/ask from unadj",
    "macro/lseg_eq_ohlc_corax.parquet::TRNOVR_UNS": "unused microstructure twin",
    "macro/lseg_eq_ohlc_corax.parquet::NUM_MOVES": "unused",
    "macro/lseg_eq_ohlc_corax.parquet::ACVOL_UNS": "volume from unadj",
    "macro/lseg_eq_ohlc_corax.parquet::VWAP": "VWAP from unadj",
    "macro/lseg_eq_ohlc_corax.parquet::VWAP_VOL": "unused",
    "macro/lseg_eq_ohlc_corax.parquet::BLKCOUNT": "unused",
    "macro/lseg_eq_ohlc_corax.parquet::BLKVOLUM": "unused",
    "macro/lseg_eq_ohlc_corax.parquet::TRD_STATUS": "QA",
    "macro/lseg_eq_ohlc_corax.parquet::ric": "identifier",
    "macro/lseg_eq_ohlc_corax.parquet::asof_ts": "provenance",
    "macro/lseg_eq_size.parquet::Free Float": "unused size field",
    "macro/lseg_eq_size.parquet::Free Float (Percent)": "unused size field",
    "macro/lseg_eq_size.parquet::ric": "identifier",
    "macro/lseg_eq_size.parquet::asof_ts": "provenance",
    "macro/lseg_gics.parquet::asof_ts": "provenance",
    "macro/p3/lseg_short_interest.parquet::ric": "identifier",
    "macro/p3/lseg_ibes.parquet::ric": "identifier",
    "macro/p3/lseg_worldscope.parquet::ric": "identifier",
    "macro/p3/lseg_worldscope.parquet::Net Income Incl Extra Before Distributions": "ni_at from Compustat preferred",
    "factors/jkp_chars.parquet::ticker": "identifier; join via permno",
    "factors/jkp_chars.parquet::citation": "provenance string",
    "macro/crsp_om_adv.parquet::permno": "link metadata; ADV keyed by secid",
    "macro/crsp_om_adv.parquet::prc": "price twin; ADV used",
    "macro/crsp_om_adv.parquet::vol": "volume twin; ADV used",
    "macro/crsp_optionm_link.parquet::score": "WRDS link quality unused at FEAT",
    "macro/ff_factors.parquet::RMW": "FF5 unused; spine uses FF4+Mom",
    "macro/ff_factors.parquet::CMA": "FF5 unused; spine uses FF4+Mom",
    "macro/pastor_stambaugh.parquet::PS_VWF": "value-weighted PS unused",
    "macro/om_distrd.parquet::record_date": "event metadata; ex_date is PIT clock",
    "macro/om_distrd.parquet::seq_num": "OM row id",
    "macro/om_distrd.parquet::adj_factor": "unused; amount used",
    "macro/om_distrd.parquet::declare_date": "event metadata",
    "macro/om_distrd.parquet::payment_date": "event metadata",
    "macro/om_distrd.parquet::link_secid": "OM metadata",
    "macro/om_distrd.parquet::distr_type": "filter metadata; all cash treated",
    "macro/om_distrd.parquet::frequency": "OM metadata",
    "macro/om_distrd.parquet::currency": "OM metadata",
    "macro/om_distrd.parquet::approx_flag": "QA",
    "macro/om_distrd.parquet::cancel_flag": "QA",
    "macro/om_distrd.parquet::liquid_flag": "QA",
    "macro/om_stdbrte.parquet::days": "tenor metadata",
    "macro/om_stdopd.parquet::forward_price": "std marks aux unused by FEAT cube",
    "macro/om_stdopd.parquet::strike_price": "std marks aux unused by FEAT cube",
    "macro/om_stdopd.parquet::premium": "std marks aux unused by FEAT cube",
    "macro/om_stdopd.parquet::gamma": "greeks unused by equity FEAT",
    "macro/om_stdopd.parquet::theta": "greeks unused by equity FEAT",
    "macro/om_stdopd.parquet::vega": "greeks unused by equity FEAT",
    "macro/om_securd.parquet::secid": "static security master",
    "macro/om_securd.parquet::cusip": "identifier",
    "macro/om_securd.parquet::ticker": "identifier",
    "macro/om_securd.parquet::sic": "identifier",
    "macro/om_securd.parquet::index_flag": "OM metadata",
    "macro/om_securd.parquet::exchange_d": "OM metadata",
    "macro/om_securd.parquet::class": "OM metadata",
    "macro/om_securd.parquet::issue_type": "OM metadata",
    "macro/om_securd.parquet::industry_group": "OM metadata",
    "macro/om_secnmd.parquet::secid": "name history master",
    "macro/om_secnmd.parquet::effect_date": "name history",
    "macro/om_secnmd.parquet::cusip": "identifier",
    "macro/om_secnmd.parquet::ticker": "identifier",
    "macro/om_secnmd.parquet::class": "OM metadata",
    "macro/om_secnmd.parquet::issuer": "OM metadata",
    "macro/om_secnmd.parquet::issue": "OM metadata",
    "macro/om_secnmd.parquet::sic": "identifier",
    "macro/compustat_funda_enrich.parquet::gvkey": "identifier",
    "macro/compustat_funda_enrich.parquet::fyear": "fiscal year label",
    "macro/compustat_funda_enrich.parquet::tic": "identifier",
    "macro/compustat_funda_enrich.parquet::cusip": "identifier",
    "macro/compustat_funda_enrich.parquet::conm": "company name",
    "macro/compustat_funda_enrich.parquet::dv": "total dividends twin; dvc used",
    "macro/compustat_funda_enrich.parquet::prcc_f": "fiscal price unused; daily close used",
    "macro/compustat_funda_enrich.parquet::csho": "shares twin; size from LSEG",
    "macro/interest_rate.parquet::lseg_rates_asof_ts": "provenance",
    "macro/lseg_spx_pit.parquet::Instrument": "index ric identifier",
    "macro/lseg_spx_pit.parquet::asof_ts": "provenance",
    "macro/sp500_hv.parquet::cusip": "identifier",
    "macro/sp500_hv.parquet::ticker": "identifier",
    "macro/sp500_hv.parquet::sic": "identifier",
    "macro/sp500_hv.parquet::index_flag": "OM metadata",
    "macro/sp500_hv.parquet::exchange_d": "OM metadata",
    "macro/sp500_hv.parquet::class": "OM metadata",
    "macro/sp500_hv.parquet::issue_type": "OM metadata",
    "macro/sp500_hv.parquet::industry_group": "OM metadata",
    "macro/spx_opvold.parquet::index_flag": "OM metadata",
    "macro/spx_opvold.parquet::cusip": "identifier",
    "macro/spx_opvold.parquet::ticker": "identifier",
    "macro/spx_opvold.parquet::sic": "identifier",
    "macro/spx_opvold.parquet::exchange_d": "OM metadata",
    "macro/spx_opvold.parquet::class": "OM metadata",
    "macro/spx_opvold.parquet::issue_type": "OM metadata",
    "macro/spx_opvold.parquet::industry_group": "OM metadata",
    # hive identifiers
    "HIVE:options_panel::symbol": "contract symbol metadata",
    "HIVE:options_panel::symbol_flag": "OM metadata",
    "HIVE:options_panel::last_date": "OM metadata",
    "HIVE:options_panel::gamma": "greeks unused by equity FEAT",
    "HIVE:options_panel::vega": "greeks unused by equity FEAT",
    "HIVE:options_panel::theta": "greeks unused by equity FEAT",
    "HIVE:options_panel::cfadj": "OM metadata",
    "HIVE:options_panel::am_settlement": "OM metadata",
    "HIVE:options_panel::contract_size": "OM metadata",
    "HIVE:options_panel::ss_flag": "OM metadata",
    "HIVE:options_panel::expiry_indicator": "OM metadata",
    "HIVE:options_panel::root": "OM metadata",
    "HIVE:options_panel::suffix": "OM metadata",
    "HIVE:options_panel::cusip": "identifier",
    "HIVE:options_panel::ticker": "identifier",
    "HIVE:options_panel::sic": "identifier",
    "HIVE:options_panel::index_flag": "OM metadata",
    "HIVE:options_panel::exchange_d": "OM metadata",
    "HIVE:options_panel::class": "OM metadata",
    "HIVE:options_panel::issue_type": "OM metadata",
    "HIVE:options_panel::industry_group": "OM metadata",
    "HIVE:options_panel::issuer": "OM metadata",
    "HIVE:options_panel::div_convention": "OM metadata",
    "HIVE:options_panel::exercise_style": "OM metadata",
    "HIVE:options_panel::am_set_flag": "OM metadata",
    "HIVE:vol_surface::cusip": "identifier",
    "HIVE:vol_surface::ticker": "identifier",
    "HIVE:vol_surface::sic": "identifier",
    "HIVE:vol_surface::index_flag": "OM metadata",
    "HIVE:vol_surface::exchange_d": "OM metadata",
    "HIVE:vol_surface::class": "OM metadata",
    "HIVE:vol_surface::issue_type": "OM metadata",
    "HIVE:vol_surface::industry_group": "OM metadata",
}


def _ibes_unused_reason(col: str) -> str:
    return f"IBES ratio not in curated FEAT-12 ({col})"


def _interest_unused_reason(col: str) -> str:
    return f"FRB rate series not in macro_loader core ({col})"


def _crsp_prices_unused_reason(col: str) -> str:
    return f"CRSP twin; equity spine uses sp500_sec OM SoT ({col})"


def _fwd_unused_reason(col: str) -> str:
    return f"forward panel aux unused by equity FEAT ({col})"


def _spx_index_unused_reason(col: str) -> str:
    return f"SPX option index dump unused by equity FEAT ({col})"


def _cboe_unused_reason(col: str) -> str:
    return f"CBOE OHLC sibling unused; close VIX used ({col})"


def _lseg_rates_unused_reason(col: str) -> str:
    return f"index vol/rates field unused by overlay (uses YLDTOMAT/FIXING_1) ({col})"


def _ric_map_unused_reason(col: str) -> str:
    return f"RIC map metadata / P4 enrichment unused by FEAT ({col})"


def _ibes_meta_unused(col: str) -> str:
    return f"IBES panel meta unused by curated FEAT ({col})"


def expand_dynamic_unused(columns_by_table: dict[str, list[str]]) -> dict[str, str]:
    """Fill INTENTIONALLY_UNUSED for large tables by convention."""
    out = dict(INTENTIONALLY_UNUSED)
    curated_ibes = {
        "public_date",
        "permno",
        "gvkey",
        "bm",
        "pe_exi",
        "ps",
        "pcf",
        "dpr",
        "npm",
        "gpm",
        "roa",
        "roe",
        "cfm",
        "evm",
        "CAPEI",
    }
    for col in columns_by_table.get("macro/ibes_financial_ratios.parquet", []):
        key = f"macro/ibes_financial_ratios.parquet::{col}"
        if key in CONSUMER_MAP or key in out:
            continue
        if col in curated_ibes:
            continue
        if col in {"adate", "qdate", "Ticker", "CUSIP"}:
            out[key] = _ibes_meta_unused(col)
        else:
            out[key] = _ibes_unused_reason(col)

    interest_used = {
        "date",
        "dtb3",
        "dgs10",
        "sofr",
        "lseg_us2y",
        "lseg_us10y",
        "lseg_sofr",
        "lseg_rates_asof_ts",
    }
    for col in columns_by_table.get("macro/interest_rate.parquet", []):
        key = f"macro/interest_rate.parquet::{col}"
        if key in CONSUMER_MAP or key in out or col in interest_used:
            continue
        out[key] = _interest_unused_reason(col)

    for col in columns_by_table.get("macro/sp500_prices.parquet", []):
        key = f"macro/sp500_prices.parquet::{col}"
        if key in CONSUMER_MAP or key in out:
            continue
        out[key] = _crsp_prices_unused_reason(col)

    for col in columns_by_table.get("macro/sp500_fwd.parquet", []):
        key = f"macro/sp500_fwd.parquet::{col}"
        if key in CONSUMER_MAP or key in out:
            continue
        out[key] = _fwd_unused_reason(col)

    for col in columns_by_table.get("macro/spx_index.parquet", []):
        key = f"macro/spx_index.parquet::{col}"
        if key in CONSUMER_MAP or key in out:
            continue
        out[key] = _spx_index_unused_reason(col)

    for col in columns_by_table.get("macro/cboe_vix.parquet", []):
        key = f"macro/cboe_vix.parquet::{col}"
        if key in CONSUMER_MAP or key in out:
            continue
        out[key] = _cboe_unused_reason(col)

    rates_used = {"date", "ric", "YLDTOMAT", "FIXING_1", "asof_ts"}
    for col in columns_by_table.get("macro/lseg_index_vol_rates.parquet", []):
        key = f"macro/lseg_index_vol_rates.parquet::{col}"
        if key in CONSUMER_MAP or key in out or col in rates_used:
            continue
        out[key] = _lseg_rates_unused_reason(col)

    ric_used = {"secid", "ric", "TR.GICSIndustry"}
    for col in columns_by_table.get("macro/lseg_ric_map.parquet", []):
        key = f"macro/lseg_ric_map.parquet::{col}"
        if key in CONSUMER_MAP or key in out or col in ric_used:
            continue
        out[key] = _ric_map_unused_reason(col)

    return out


def _sample_hive_schema(lake: Path, name: str) -> list[str]:
    root = lake / name
    if not root.is_dir():
        return []
    years = sorted(root.glob("year=*"))
    if not years:
        return []
    months = sorted(years[0].glob("month=*"))
    files = list(months[0].glob("*.parquet")) if months else list(years[0].rglob("*.parquet"))
    if not files:
        return []
    return list(pq.read_schema(files[0]).names)


def collect_lake_columns(lake: Path | None = None) -> dict[str, list[str]]:
    root = assert_lake_mounted(lake) if lake is None else Path(lake)
    out: dict[str, list[str]] = {}
    for rel in FLAT_TABLES:
        path = root / rel
        if not path.is_file():
            continue
        out[rel] = list(pq.read_schema(path).names)
    for hive in HIVE_TABLES:
        cols = _sample_hive_schema(root, hive)
        if cols:
            out[f"HIVE:{hive}"] = cols
    return out


def run_coverage_audit(
    lake: Path | None = None,
    *,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Map every lake column to CONSUMER_MAP or INTENTIONALLY_UNUSED; fail on gaps."""
    root = assert_lake_mounted(lake) if lake is None else Path(lake)
    columns_by_table = collect_lake_columns(root)
    unused = expand_dynamic_unused(columns_by_table)
    classified: dict[str, str] = {}
    missing: list[str] = []
    collisions: list[str] = []
    for table, cols in sorted(columns_by_table.items()):
        for col in cols:
            key = f"{table}::{col}" if not table.startswith("HIVE:") else f"{table}::{col}"
            in_c = key in CONSUMER_MAP
            in_u = key in unused
            if in_c and in_u:
                collisions.append(key)
            elif in_c:
                classified[key] = f"consumer:{CONSUMER_MAP[key]}"
            elif in_u:
                classified[key] = f"unused:{unused[key]}"
            else:
                missing.append(key)
    report = {
        "lake": str(root),
        "n_tables": len(columns_by_table),
        "n_columns": sum(len(v) for v in columns_by_table.values()),
        "n_classified": len(classified),
        "n_missing": len(missing),
        "n_collisions": len(collisions),
        "missing": missing,
        "collisions": collisions,
        "classified": classified,
        "ok": not missing and not collisions,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def assert_coverage_ok(lake: Path | None = None) -> dict[str, Any]:
    report = run_coverage_audit(lake)
    if report["missing"]:
        sample = report["missing"][:20]
        raise AssertionError(
            f"unclassified lake columns ({len(report['missing'])}): {sample}"
        )
    if report["collisions"]:
        raise AssertionError(f"columns in both CONSUMER_MAP and UNUSED: {report['collisions'][:20]}")
    return report
