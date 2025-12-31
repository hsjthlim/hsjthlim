# ============================================================
# Tech-Overlay Strategy Validation Engine (REFACTORED)
# ============================================================
# - Strategy validation engine + inspection dashboard
# - Clean separation of concerns with unidirectional dependencies
# - Single entry: run_pipeline()
# ============================================================

import os
import datetime
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
import inspect

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pandas.errors import ParserError


# ============================================================
# 00. CONFIG
# ============================================================

START_DATE = "1998-12-31"
END_DATE = datetime.datetime.today().strftime("%Y-%m-%d")
DATA_PATH = None
REPORT_DIR = r"C:\Users\User\Desktop\py\sector\report_new"

SECTOR_ETFS = {
    "XLK": "Tech", "XLF": "Financials", "XLV": "Health", "XLE": "Energy",
    "XLI": "Industrials", "XLY": "ConsDisc", "XLP": "ConsStap", "XLU": "Utilities",
    "XLB": "Materials", "XLRE": "RealEstate", "XLC": "CommSvc", "SPY": "SPX",
}

TECH_COL = "XLK"
MARKET_COL = "SPY"

REBALANCE_FREQ_DEFAULT = "2W"
PERIODS_PER_YEAR_MAP = {"M": 12, "W": 52, "2W": 26}

BURN_IN_PERIODS = 26
IS_PERIODS = 130
OOS_PERIODS = 26

WINDOW_K_PERIODS = 52
PERCENTILE_LOOKBACK = 52
MOM_WINDOW_PERIODS = 26
CORR_WINDOW_PERIODS = 52

PERCENTILE_LOW_GRID_DEFAULT = [20, 30, 40]
PERCENTILE_HIGH_GRID_DEFAULT = [60, 70, 80]

P_FIXED = 0.50

BASE_N_NON_TECH = 4
BASE_N_DELTA = 2
W_PER_SECTOR = 0.10

ENABLE_OFF_SWITCH_DEFAULT = True
OFF_LOOKBACK_DEFAULT = 4
OFF_MDD_THRESHOLD_DEFAULT = -0.10
OFF_COOLDOWN_DEFAULT = 1

BOOTSTRAP_BLOCK_MONTHS_DEFAULT = 2
BOOTSTRAP_BLOCK_SIZE_OVERRIDE_DEFAULT = None
N_BOOTSTRAP_DEFAULT = 1000
N_PERM_DEFAULT = 1000

EXPERIMENT_MODE = False

_SNAPSHOT_BASELINE = {
    "operation_sharpe": None,
    "operation_ir": None,
    "operation_mdd": None,
    "pct_low": None,
    "pct_high": None,
    "operation_cum_last": None,
}

# ============================================================
# 01. DATA STRUCTURES
# ============================================================

@dataclass
class DataBundle:
    """Data Engine의 출력"""
    prices: pd.DataFrame
    bin_last_obs: pd.Series
    freq: str
    periods_per_year: int
    daily_prices: pd.DataFrame = None


@dataclass
class SignalBundle:
    """Signal Engine의 출력"""
    tech_resid: pd.Series
    tech_z: pd.Series
    tech_pct: pd.Series
    mom: pd.DataFrame
    corr: pd.DataFrame
    prices: pd.DataFrame
    returns: pd.DataFrame


@dataclass
class StrategyParams:
    """전략 파라미터"""
    pct_low: float
    pct_high: float
    p: float = P_FIXED
    base_n: int = BASE_N_NON_TECH
    delta_n: int = BASE_N_DELTA
    w_per_sector: float = W_PER_SECTOR
    tech_col: str = TECH_COL
    market_col: str = MARKET_COL


@dataclass
class PipelineResult:
    """Pipeline 전체 출력"""
    freq: str
    periods_per_year: int
    data: DataBundle
    signals: SignalBundle
    
    oos_evaluation: Dict[str, Any]
    operation: Dict[str, Any]
    
    operation_sharpe: float
    operation_ir: float
    operation_mdd: float
    
    confirmed_regime: Dict[str, Any]
    live_regime: Dict[str, Any]
    live_position: Dict[str, Any]
    
    sector_analysis: Dict[str, Any]
    regime_perf: Dict[str, Any]
    metrics: Dict[str, Any]
    
    bootstrap: Optional[Dict[str, Any]] = None
    permutation: Optional[Dict[str, Any]] = None
    off_compare: Optional[Dict[str, Any]] = None
    
    ui_params: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 02. UTILITIES (PURE FUNCTIONS)
# ============================================================

def calc_returns(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change(fill_method=None)


def calc_cumulative(ret: pd.Series) -> pd.Series:
    return (1.0 + ret).cumprod()


def calc_sharpe(ret: pd.Series, periods_per_year: int) -> float:
    ret = ret.dropna()
    if ret.empty:
        return np.nan
    sd = ret.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return (ret.mean() / sd) * np.sqrt(periods_per_year)


def calc_mdd(cum: pd.Series) -> float:
    cum = cum.dropna()
    if cum.empty:
        return np.nan
    return (cum / cum.cummax() - 1.0).min()


def calc_ir(port: pd.Series, bench: pd.Series, periods_per_year: int) -> float:
    excess = (port - bench).dropna()
    if len(excess) < 2:
        return np.nan
    te = excess.std(ddof=1)
    if te == 0 or not np.isfinite(te):
        return np.nan
    return (excess.mean() * periods_per_year) / (te * np.sqrt(periods_per_year))


def minmax_norm_series(s: pd.Series) -> pd.Series:
    if s is None or s.empty or s.isna().all():
        return s
    smin, smax = s.min(), s.max()
    if np.isclose(smax, smin):
        return pd.Series(0.5, index=s.index)
    return (s - smin) / (smax - smin)


def periods_per_year_from_freq(freq: str) -> int:
    if freq not in PERIODS_PER_YEAR_MAP:
        raise ValueError(f"Unknown freq: {freq}")
    return PERIODS_PER_YEAR_MAP[freq]


def infer_block_size(freq: str, block_months: int, override: Optional[int] = None) -> int:
    if override is not None:
        return int(max(2, override))
    ppy = periods_per_year_from_freq(freq)
    per_month = ppy / 12.0
    size = int(np.round(block_months * per_month))
    return int(max(2, size))


def fmt_pct(x) -> str:
    try:
        if x is None:
            return "N/A"
        if isinstance(x, (pd.Timestamp, datetime.date, datetime.datetime, np.datetime64)):
            return "N/A"
        if not np.isfinite(float(x)):
            return "N/A"
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "N/A"


def fmt_num(x, digits=3) -> str:
    try:
        if x is None:
            return "N/A"
        if isinstance(x, (pd.Timestamp, datetime.date, datetime.datetime, np.datetime64)):
            return "N/A"
        v = float(x)
        if not np.isfinite(v):
            return "N/A"
        return f"{v:.{digits}f}"
    except Exception:
        return "N/A"


def compute_portfolio_returns(weights: pd.DataFrame, df_prices: pd.DataFrame) -> pd.Series:
    ret = calc_returns(df_prices)
    return (weights.shift(1) * ret).sum(axis=1)


def _extract_snapshot_baseline(bundle: Dict) -> dict:
    last_param = bundle["oos_evaluation"]["param_history"].iloc[-1]
    cum = bundle["operation"]["operation_cumulative"]
    return {
        "operation_sharpe": float(bundle["operation"]["operation_sharpe"]),
        "operation_ir": float(bundle["operation"]["operation_ir"]),
        "operation_mdd": float(bundle["operation"]["operation_mdd"]),
        "pct_low": float(last_param["pct_low"]),
        "pct_high": float(last_param["pct_high"]),
        "operation_cum_last": float(cum.iloc[-1]) if len(cum) else np.nan,
    }


def _assert_frozen_baseline(bundle: Dict, tol: float = 1e-10) -> None:
    if EXPERIMENT_MODE:
        return
    ref = _SNAPSHOT_BASELINE
    if ref["operation_sharpe"] is None:
        return

    def close(a, b, t=tol):
        return abs(float(a) - float(b)) <= t

    op = bundle["operation"]
    assert close(op["operation_sharpe"], ref["operation_sharpe"]), "❌ Sharpe drift"
    assert close(op["operation_ir"], ref["operation_ir"]), "❌ IR drift"
    assert close(op["operation_mdd"], ref["operation_mdd"]), "❌ MDD drift"
    assert close(op["operation_cum_last"], ref["operation_cum_last"]), "❌ Cumulative signature drift"

    last = bundle["oos_evaluation"]["param_history"].iloc[-1]
    assert close(last["pct_low"], ref["pct_low"], 0.0), "❌ pct_low changed"
    assert close(last["pct_high"], ref["pct_high"], 0.0), "❌ pct_high changed"

    # ============================================================
# 03. DATA ENGINE
# ============================================================

def download_price_data() -> pd.DataFrame:
    series_list = []
    for ticker in SECTOR_ETFS.keys():
        try:
            raw = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
            if not raw.empty:
                s = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
                s.name = ticker
                series_list.append(s)
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")

    if not series_list:
        raise RuntimeError("No data downloaded from yfinance.")

    df = pd.concat(series_list, axis=1).sort_index()
    return df


def load_data() -> pd.DataFrame:
    return download_price_data()


def resample_to_period(df_daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "W":
        return df_daily.resample("W-FRI").last()
    if freq == "M":
        return df_daily.resample("M").last()
    if freq == "2W":
        weekly = df_daily.resample("W-FRI").last()
        return weekly.iloc[::2].copy()
    raise ValueError(f"Unknown REBALANCE_FREQ: {freq}")


def map_bin_to_last_daily_obs(df_daily: pd.DataFrame, bin_ends: pd.DatetimeIndex) -> pd.Series:
    """
    For each resampled bin-end label, return the last daily observation date <= bin_end
    and > previous bin_end.
    """
    if df_daily.empty or len(bin_ends) == 0:
        return pd.Series(dtype="datetime64[ns]")

    daily_idx = df_daily.index
    out = []
    prev_end = None
    for be in bin_ends:
        if prev_end is None:
            mask = (daily_idx <= be)
        else:
            mask = (daily_idx > prev_end) & (daily_idx <= be)
        dates_in_bin = daily_idx[mask]
        out.append(dates_in_bin.max() if len(dates_in_bin) else pd.NaT)
        prev_end = be
    return pd.Series(out, index=bin_ends, name="last_daily_obs")


def prepare_data(freq: str) -> DataBundle:
    """
    Data Engine의 단일 진입점
    모든 데이터 준비 과정을 캡슐화
    """
    df_daily = load_data()
    prices = resample_to_period(df_daily, freq)
    bin_last_obs = map_bin_to_last_daily_obs(df_daily, prices.index)
    periods_per_year = periods_per_year_from_freq(freq)
    
    return DataBundle(
        prices=prices,
        bin_last_obs=bin_last_obs,
        freq=freq,
        periods_per_year=periods_per_year,
        daily_prices=df_daily
    )


# ============================================================
# 04. SIGNAL ENGINE
# ============================================================

def compute_tech_residuals(df: pd.DataFrame, window: int = WINDOW_K_PERIODS,
                          tech_col: str = TECH_COL, market_col: str = MARKET_COL) -> pd.Series:
    tech = np.log(df[tech_col].replace(0, np.nan))
    mkt = np.log(df[market_col].replace(0, np.nan))
    residuals = pd.Series(index=df.index, dtype=float)

    for i in range(window - 1, len(df)):
        y = tech.iloc[i-window+1:i+1].values
        x = mkt.iloc[i-window+1:i+1].values
        if np.isnan(y).any() or np.isnan(x).any():
            continue
        try:
            beta, alpha = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
            residuals.iloc[i] = y[-1] - (alpha + beta * x[-1])
        except Exception:
            pass

    return residuals


def compute_tech_zscore(residuals: pd.Series, window: int = WINDOW_K_PERIODS) -> pd.Series:
    mu = residuals.rolling(window).mean()
    sd = residuals.rolling(window).std(ddof=1)
    return (residuals - mu) / sd


def compute_tech_percentile(tech_z: pd.Series, lookback: int = PERCENTILE_LOOKBACK) -> pd.Series:
    percentiles = pd.Series(index=tech_z.index, dtype=float)
    for i in range(lookback, len(tech_z)):
        w = tech_z.iloc[i-lookback:i]
        if w.dropna().empty:
            continue
        percentiles.iloc[i] = (w < tech_z.iloc[i]).mean() * 100.0
    return percentiles


def compute_momentum(df: pd.DataFrame, window: int = MOM_WINDOW_PERIODS) -> pd.DataFrame:
    return df / df.shift(window) - 1.0


def compute_corr_with_tech(df_ret: pd.DataFrame, window: int = CORR_WINDOW_PERIODS,
                           tech_col: str = TECH_COL) -> pd.DataFrame:
    corr_df = pd.DataFrame(index=df_ret.index, columns=df_ret.columns, dtype=float)
    tech_ret = df_ret[tech_col]
    for col in df_ret.columns:
        if col == tech_col:
            corr_df[col] = 1.0
        else:
            corr_df[col] = df_ret[col].rolling(window).corr(tech_ret)
    return corr_df


def compute_signals(data: DataBundle) -> SignalBundle:
    """
    Signal Engine의 단일 진입점
    DataBundle만 참조하여 모든 시그널 계산
    """
    df = data.prices
    df_ret = calc_returns(df)
    
    tech_resid = compute_tech_residuals(df)
    tech_z = compute_tech_zscore(tech_resid)
    tech_pct = compute_tech_percentile(tech_z)
    mom = compute_momentum(df)
    corr = compute_corr_with_tech(df_ret)
    
    return SignalBundle(
        tech_resid=tech_resid,
        tech_z=tech_z,
        tech_pct=tech_pct,
        mom=mom,
        corr=corr,
        prices=df,
        returns=df_ret
    )

# ============================================================
# 05. STRATEGY ENGINE
# ============================================================

def decide_n_sectors(
    pct: float,
    pct_low: float,
    pct_high: float,
    base_n: int = BASE_N_NON_TECH,
    delta_n: int = BASE_N_DELTA
) -> int:
    if not np.isfinite(pct):
        return base_n
    if pct <= pct_low:
        return max(base_n - delta_n, 0)
    if pct >= pct_high:
        return base_n + delta_n
    return base_n


def compute_regime(tech_pct: float, pct_low: float, pct_high: float) -> str:
    if not np.isfinite(tech_pct):
        return "Unknown"
    if tech_pct <= pct_low:
        return "Tech Depressed"
    if tech_pct >= pct_high:
        return "Tech Overheated"
    return "Tech Neutral"


class WeightGenerator:
    """
    전략 로직 캡슐화
    SignalBundle과 파라미터만 참조하여 가중치 생성
    OOS와 Operation에서 재사용 가능
    """
    
    def __init__(self, signals: SignalBundle, params: StrategyParams):
        self.signals = signals
        self.params = params
    
    def generate_weights_for_date(self, date) -> Optional[pd.Series]:
        """특정 날짜의 타겟 가중치 생성"""
        if date not in self.signals.tech_pct.index:
            return None
        
        pct_val = self.signals.tech_pct.loc[date]
        n_sectors = decide_n_sectors(
            pct_val, 
            self.params.pct_low, 
            self.params.pct_high,
            self.params.base_n,
            self.params.delta_n
        )
        
        cols = list(self.signals.prices.columns)
        w = pd.Series(0.0, index=cols)
        
        if n_sectors <= 0:
            w[self.params.tech_col] = 1.0
            return w
        
        non_tech_cols = [c for c in cols if c not in (self.params.tech_col, self.params.market_col)]
        mom_raw = self.signals.mom.loc[date, non_tech_cols]
        mom_norm = minmax_norm_series(mom_raw)
        
        corr_raw = self.signals.corr.loc[date, non_tech_cols]
        corr_score = 1.0 - (corr_raw + 1.0) / 2.0
        
        scores = (self.params.p * mom_norm + (1.0 - self.params.p) * corr_score).dropna()
        if scores.empty:
            w[self.params.tech_col] = 1.0
            return w
        
        top = scores.sort_values(ascending=False).head(n_sectors)
        w_each = min(self.params.w_per_sector, 0.9 / len(top))
        w_tech = max(1.0 - w_each * len(top), 0.0)
        
        w[self.params.tech_col] = w_tech
        w[top.index] = w_each
        return w
    
    def generate_weights(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """여러 날짜의 가중치를 한번에 생성"""
        weights_list = []
        for date in dates:
            w = self.generate_weights_for_date(date)
            weights_list.append(w)
        return pd.DataFrame(weights_list, index=dates)


# ============================================================
# 06. OOS EVALUATION ENGINE
# ============================================================

def run_is_oos(
    data: DataBundle,
    signals: SignalBundle,
    pct_low_grid: List[int],
    pct_high_grid: List[int],
) -> dict:
    """
    OOS 평가 엔진
    DataBundle과 SignalBundle만 참조
    """
    prices = data.prices
    dates = prices.index
    n = len(dates)
    oos_returns_list = []
    oos_weights_list = []
    param_history = []
    
    bench_ret_all = calc_returns(prices[[MARKET_COL]])[MARKET_COL]
    
    for is_start in range(BURN_IN_PERIODS, n - (IS_PERIODS + OOS_PERIODS) + 1, OOS_PERIODS):
        is_end = is_start + IS_PERIODS
        oos_start, oos_end = is_end, is_end + OOS_PERIODS
        
        is_idx = dates[is_start:is_end]
        oos_idx = dates[oos_start:oos_end]
        
        spy_ret_is = bench_ret_all.loc[is_idx].dropna()
        
        best = None
        for lo in pct_low_grid:
            for hi in pct_high_grid:
                if hi <= lo:
                    continue
                
                params = StrategyParams(pct_low=lo, pct_high=hi)
                weight_gen = WeightGenerator(signals, params)
                w_is = weight_gen.generate_weights(is_idx)
                
                r_is = compute_portfolio_returns(w_is, prices.loc[is_idx]).dropna()
                ir_is = calc_ir(r_is, spy_ret_is.reindex(r_is.index), periods_per_year=data.periods_per_year)
                
                if not np.isfinite(ir_is):
                    continue
                
                cand = {
                    "pct_low": lo, "pct_high": hi,
                    "ir": ir_is,
                    "sharpe": calc_sharpe(r_is, periods_per_year=data.periods_per_year),
                    "mdd": calc_mdd(calc_cumulative(r_is)),
                }
                if best is None or cand["ir"] > best["ir"]:
                    best = cand
        
        if best is None:
            continue
        
        param_history.append({
            "IS_start": is_idx[0], "IS_end": is_idx[-1],
            "OOS_start": oos_idx[0], "OOS_end": oos_idx[-1],
            **best
        })
        
        # OOS 기간 가중치 생성 (best params 사용)
        best_params = StrategyParams(pct_low=best["pct_low"], pct_high=best["pct_high"])
        weight_gen = WeightGenerator(signals, best_params)
        w_oos = weight_gen.generate_weights(oos_idx)
        r_oos = compute_portfolio_returns(w_oos, prices.loc[oos_idx])
        
        oos_returns_list.append(r_oos)
        oos_weights_list.append(w_oos)
    
    if not oos_returns_list:
        raise RuntimeError("OOS results are empty. Check data length or parameter settings.")
    
    oos_ret = pd.concat(oos_returns_list).sort_index()
    oos_weights = pd.concat(oos_weights_list).sort_index()
    
    spy_ret = bench_ret_all.reindex(oos_ret.index).dropna()
    oos_ret = oos_ret.reindex(spy_ret.index)
    
    return {
        "oos_return_raw": oos_ret,
        "oos_weights": oos_weights,
        "spy_return": spy_ret,
        "param_history": pd.DataFrame(param_history),
    }

    # ============================================================
# 07. OPERATION ENGINE
# ============================================================

def apply_off_switch(
    strat_ret: pd.Series,
    bench_ret: pd.Series,
    enable_off: bool,
    off_lookback: int,
    off_mdd_threshold: float,
    off_cooldown: int
):
    if not enable_off:
        return pd.Series(False, index=strat_ret.index), strat_ret
    
    cum = calc_cumulative(strat_ret)
    off_mask = pd.Series(False, index=strat_ret.index)
    
    i = off_lookback
    while i < len(strat_ret):
        window = cum.iloc[i-off_lookback:i+1]
        mdd = calc_mdd(window)
        if np.isfinite(mdd) and mdd <= off_mdd_threshold:
            end_idx = min(i + off_cooldown, len(strat_ret))
            off_mask.iloc[i:end_idx] = True
            i = end_idx
        else:
            i += 1
    
    final_ret = strat_ret.copy()
    final_ret.loc[off_mask] = bench_ret.loc[off_mask]
    return off_mask, final_ret


def compute_operation_returns(
    data: DataBundle,
    signals: SignalBundle,
    param_history: pd.DataFrame,
    enable_off: bool,
    off_lookback: int,
    off_mdd_threshold: float,
    off_cooldown: int
) -> dict:
    """
    Operation Engine
    확정된 파라미터로 closed bin만 처리
    WeightGenerator를 사용하여 전략 로직 재사용
    """
    prices = data.prices
    all_dates = prices.index
    operation_returns_list = []
    operation_weights_list = []
    bench_ret_all = calc_returns(prices[[MARKET_COL]])[MARKET_COL]
    
    if param_history is None or param_history.empty:
        raise RuntimeError("param_history is empty. Cannot compute operation returns.")
    
    for _, row in param_history.iterrows():
        oos_end = pd.Timestamp(row["OOS_end"])
        lo, hi = float(row["pct_low"]), float(row["pct_high"])
        
        if oos_end not in all_dates:
            continue
        
        oos_end_idx = int(all_dates.get_loc(oos_end))
        next_bin_start = oos_end_idx + 1
        next_bin_end = min(next_bin_start + OOS_PERIODS, len(all_dates))
        
        if next_bin_start >= len(all_dates):
            continue
        
        bin_idx = all_dates[next_bin_start:next_bin_end]
        
        # WeightGenerator 사용하여 가중치 생성
        params = StrategyParams(pct_low=lo, pct_high=hi)
        weight_gen = WeightGenerator(signals, params)
        w_bin = weight_gen.generate_weights(bin_idx)
        r_bin = compute_portfolio_returns(w_bin, prices.loc[bin_idx])
        
        operation_returns_list.append(r_bin)
        operation_weights_list.append(w_bin)
    
    if not operation_returns_list:
        raise RuntimeError("No operation returns generated.")
    
    op_ret_raw = pd.concat(operation_returns_list).sort_index()
    op_weights = pd.concat(operation_weights_list).sort_index()
    
    op_spy = bench_ret_all.reindex(op_ret_raw.index).dropna()
    op_ret_raw = op_ret_raw.reindex(op_spy.index)
    
    off_mask, op_ret_final = apply_off_switch(
        op_ret_raw, op_spy, enable_off, off_lookback, off_mdd_threshold, off_cooldown
    )
    
    return {
        "operation_return_raw": op_ret_raw,
        "operation_return": op_ret_final,
        "operation_weights": op_weights,
        "operation_spy_return": op_spy,
        "operation_off_mask": off_mask,
        "operation_cumulative": calc_cumulative(op_ret_final),
        "operation_spy_cumulative": calc_cumulative(op_spy),
    }


# ============================================================
# 08. ANALYSIS ENGINE
# ============================================================

def classify_regime(tech_pct: pd.Series, pct_low: float, pct_high: float) -> pd.Series:
    regime = pd.Series(index=tech_pct.index, dtype=str)
    regime.loc[tech_pct <= pct_low] = "Tech Depressed"
    regime.loc[(tech_pct > pct_low) & (tech_pct < pct_high)] = "Tech Neutral"
    regime.loc[tech_pct >= pct_high] = "Tech Overheated"
    return regime


def analyze_sectors(
    weights: pd.DataFrame,
    tech_pct: pd.Series,
    pct_low: float,
    pct_high: float,
) -> dict:
    non_tech_cols = [c for c in weights.columns if c not in (TECH_COL, MARKET_COL)]
    regime = classify_regime(tech_pct, pct_low, pct_high)
    
    selection_freq = (weights[non_tech_cols] > 0).sum()
    regime_selection = {r: (weights.loc[regime == r, non_tech_cols] > 0).sum()
                        for r in regime.dropna().unique()}
    
    return {
        "selection_freq": selection_freq,
        "regime_selection": regime_selection,
        "avg_n_sectors": (weights[non_tech_cols] > 0).sum(axis=1).mean(),
        "regime": regime,
    }


# ============================================================
# 09. BOOTSTRAP / PERMUTATION VALIDATION
# ============================================================

def _aligned_returns(oos_ret: pd.Series, spy_ret: pd.Series):
    df = pd.concat([oos_ret.rename("strat"), spy_ret.rename("spy")], axis=1).dropna()
    return df["strat"], df["spy"]


def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator):
    if n <= 0:
        return np.array([], dtype=int)
    block_size = int(max(1, block_size))
    n_blocks = int(np.ceil(n / block_size))
    
    idx = []
    max_start = max(1, n - block_size + 1)
    for _ in range(n_blocks):
        s = int(rng.integers(0, max_start))
        e = min(s + block_size, n)
        idx.extend(range(s, e))
    return np.array(idx[:n], dtype=int)


def bootstrap_analysis(
    oos_ret: pd.Series,
    spy_ret: pd.Series,
    periods_per_year: int,
    n_bootstrap: int,
    block_size: int,
    seed: int = 42
) -> dict:
    strat, spy = _aligned_returns(oos_ret, spy_ret)
    n = len(strat)
    if n < 10:
        return {"ci": None, "dist": None, "meta": {"n": n, "block_size": block_size}}
    
    rng = np.random.default_rng(seed)
    
    orig_sharpe = calc_sharpe(strat, periods_per_year)
    orig_ir = calc_ir(strat, spy, periods_per_year)
    orig_mdd = calc_mdd(calc_cumulative(strat))
    orig_hit = (strat > 0).mean()
    
    dist = {"sharpe": [], "ir": [], "mdd": [], "hit_ratio": []}
    
    for _ in range(int(n_bootstrap)):
        idx = block_bootstrap_indices(n, block_size, rng)
        s_s = strat.iloc[idx].reset_index(drop=True)
        s_b = spy.iloc[idx].reset_index(drop=True)
        
        dist["sharpe"].append(calc_sharpe(s_s, periods_per_year))
        dist["ir"].append(calc_ir(s_s, s_b, periods_per_year))
        dist["mdd"].append(calc_mdd(calc_cumulative(s_s)))
        dist["hit_ratio"].append((s_s > 0).mean())
    
    ci = {}
    for k, vals in dist.items():
        vals = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        if len(vals) == 0:
            ci[k] = None
            continue
        
        original = {"sharpe": orig_sharpe, "ir": orig_ir, "mdd": orig_mdd, "hit_ratio": orig_hit}[k]
        ci[k] = {
            "original": float(original) if np.isfinite(original) else np.nan,
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
        }
    
    return {
        "ci": ci,
        "dist": dist,
        "meta": {"n": int(n), "block_size": int(block_size), "n_bootstrap": int(n_bootstrap)}
    }


def permutation_test_ir(
    oos_ret: pd.Series,
    spy_ret: pd.Series,
    periods_per_year: int,
    n_perm: int,
    seed: int = 7
) -> Optional[dict]:
    strat, spy = _aligned_returns(oos_ret, spy_ret)
    excess = (strat - spy).dropna()
    if len(excess) < 10:
        return None
    
    rng = np.random.default_rng(seed)
    observed_ir = calc_ir(strat, spy, periods_per_year)
    
    null_irs = []
    e = excess.values.astype(float)
    for _ in range(int(n_perm)):
        signs = rng.choice([-1.0, 1.0], size=len(e))
        pe = e * signs
        sd = np.std(pe, ddof=1)
        if sd <= 0 or not np.isfinite(sd):
            continue
        ir = (np.mean(pe) * periods_per_year) / (sd * np.sqrt(periods_per_year))
        null_irs.append(float(ir))
    
    if len(null_irs) == 0:
        return None
    
    null_irs = np.array(null_irs, dtype=float)
    p_value = float(np.mean(np.abs(null_irs) >= np.abs(observed_ir)))
    
    return {
        "observed_ir": float(observed_ir) if np.isfinite(observed_ir) else np.nan,
        "p_value": p_value,
        "null_distribution": null_irs,
        "n_perm": int(n_perm),
        "n_used": int(len(null_irs))
    }


def bootstrap_off_comparison(
    oos_ret_raw: pd.Series,
    oos_ret_off: pd.Series,
    spy_ret: pd.Series,
    periods_per_year: int,
    n_bootstrap: int,
    block_size: int,
    seed: int = 42
) -> Optional[dict]:
    strat_raw, spy = _aligned_returns(oos_ret_raw, spy_ret)
    strat_off, _ = _aligned_returns(oos_ret_off, spy_ret)
    
    n = min(len(strat_raw), len(strat_off), len(spy))
    strat_raw = strat_raw.iloc[:n]
    strat_off = strat_off.iloc[:n]
    spy = spy.iloc[:n]
    
    if n < 10:
        return None
    
    rng = np.random.default_rng(seed)
    
    dist = {
        "sharpe_raw": [],
        "sharpe_off": [],
        "mdd_raw": [],
        "mdd_off": [],
    }
    
    for _ in range(int(n_bootstrap)):
        idx = block_bootstrap_indices(n, block_size, rng)
        
        s_raw = strat_raw.iloc[idx].reset_index(drop=True)
        s_off = strat_off.iloc[idx].reset_index(drop=True)
        
        dist["sharpe_raw"].append(calc_sharpe(s_raw, periods_per_year))
        dist["sharpe_off"].append(calc_sharpe(s_off, periods_per_year))
        
        dist["mdd_raw"].append(calc_mdd(calc_cumulative(s_raw)))
        dist["mdd_off"].append(calc_mdd(calc_cumulative(s_off)))
    
    out = {k: np.array(v, dtype=float) for k, v in dist.items()}
    out["meta"] = {"n": int(n), "block_size": int(block_size), "n_bootstrap": int(n_bootstrap)}
    return out



# ============================================================
# 10. PIPELINE (SINGLE ENTRY POINT)
# ============================================================

def run_pipeline(
    freq: str,
    pct_low_grid: List[int],
    pct_high_grid: List[int],
    enable_off: bool,
    off_lookback: int,
    off_mdd_threshold: float,
    off_cooldown: int,
    run_bootstrap: bool,
    bootstrap_months: int,
    bootstrap_override: Optional[int],
    n_bootstrap: int,
    n_perm: int,
) -> PipelineResult:
    """
    Pipeline의 단일 진입점
    각 엔진을 순차적으로 호출하며, 각 엔진은 이전 단계의 Bundle만 참조
    """
    
    # 1. Data Engine
    data = prepare_data(freq)
    
    # 2. Signal Engine
    signals = compute_signals(data)
    
    # 3. OOS Evaluation
    oos_eval = run_is_oos(
        data=data,
        signals=signals,
        pct_low_grid=pct_low_grid,
        pct_high_grid=pct_high_grid,
    )
    
    param_history = oos_eval["param_history"]
    latest_params = param_history.iloc[-1]
    lo_used = float(latest_params["pct_low"])
    hi_used = float(latest_params["pct_high"])
    last_oos_end = pd.Timestamp(latest_params["OOS_end"])
    
    confirmed_pct = signals.tech_pct.loc[last_oos_end] if last_oos_end in signals.tech_pct.index else np.nan
    confirmed_regime = compute_regime(confirmed_pct, lo_used, hi_used)
    
    confirmed = {
        "date": last_oos_end,
        "percentile": confirmed_pct,
        "regime": confirmed_regime,
        "pct_low_used": lo_used,
        "pct_high_used": hi_used,
    }
    
    # 4. Operation Engine
    op = compute_operation_returns(
        data=data,
        signals=signals,
        param_history=param_history,
        enable_off=enable_off,
        off_lookback=off_lookback,
        off_mdd_threshold=off_mdd_threshold,
        off_cooldown=off_cooldown
    )
    
    op_ret = op["operation_return"]
    op_spy = op["operation_spy_return"]
    
    # Live regime/position (last completed bin)
    live_asof = op_ret.index[-1]
    live_pct = signals.tech_pct.loc[live_asof] if live_asof in signals.tech_pct.index else np.nan
    live_reg = compute_regime(live_pct, lo_used, hi_used)
    
    live_snapshot = {
        "date": live_asof,
        "last_obs": data.bin_last_obs.get(live_asof, pd.NaT),
        "percentile": live_pct,
        "regime": live_reg,
    }
    
    # Live position weights
    params = StrategyParams(pct_low=lo_used, pct_high=hi_used)
    weight_gen = WeightGenerator(signals, params)
    live_w = weight_gen.generate_weights_for_date(live_asof)
    live_w_pos = live_w[live_w > 0].sort_values(ascending=False) if live_w is not None else pd.Series(dtype=float)
    
    live_position = {
        "date": live_asof,
        "last_obs": data.bin_last_obs.get(live_asof, pd.NaT),
        "percentile": live_pct,
        "regime": live_reg,
        "weights": live_w_pos.to_dict(),
        "pct_low_used": lo_used,
        "pct_high_used": hi_used,
    }
    
    # Metrics
    n_years = (op_ret.index[-1] - op_ret.index[0]).days / 365.25
    total_ret = op["operation_cumulative"].iloc[-1] - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
    ann_vol = op_ret.std(ddof=1) * np.sqrt(data.periods_per_year)
    hit_ratio = (op_ret > 0).mean()
    
    spy_total = op["operation_spy_cumulative"].iloc[-1] - 1.0
    spy_cagr = (1.0 + spy_total) ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan
    spy_vol = op_spy.std(ddof=1) * np.sqrt(data.periods_per_year)
    spy_hit = (op_spy > 0).mean()
    
    op_sharpe = calc_sharpe(op_ret, data.periods_per_year)
    op_ir = calc_ir(op_ret, op_spy, data.periods_per_year)
    op_mdd = calc_mdd(op["operation_cumulative"])
    
    # Sector analysis
    tech_pct_op = signals.tech_pct.reindex(op_ret.index)
    sector_analysis = analyze_sectors(op["operation_weights"], tech_pct_op, lo_used, hi_used)
    
    # Regime performance
    regime = sector_analysis["regime"]
    regime_perf = {}
    for r in ["Tech Depressed", "Tech Neutral", "Tech Overheated"]:
        mask = (regime == r)
        if mask.sum() > 0:
            strat_ret_r = op_ret.loc[mask].dropna()
            spy_ret_r = op_spy.reindex(strat_ret_r.index).dropna()
            if not strat_ret_r.empty and not spy_ret_r.empty:
                regime_perf[r] = {
                    "count": int(mask.sum()),
                    "strat_sharpe": calc_sharpe(strat_ret_r, data.periods_per_year),
                    "spy_sharpe": calc_sharpe(spy_ret_r, data.periods_per_year),
                    "strat_cum": (1.0 + strat_ret_r).prod() - 1.0,
                    "spy_cum": (1.0 + spy_ret_r).prod() - 1.0,
                }
    
    # Bootstrap validation
    bootstrap_res = None
    perm_res = None
    off_compare = None
    block_size = infer_block_size(freq, bootstrap_months, bootstrap_override)
    
    if run_bootstrap:
        bootstrap_res = bootstrap_analysis(
            oos_ret=op_ret,
            spy_ret=op_spy,
            periods_per_year=data.periods_per_year,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            seed=42
        )
        perm_res = permutation_test_ir(
            oos_ret=op_ret,
            spy_ret=op_spy,
            periods_per_year=data.periods_per_year,
            n_perm=n_perm,
            seed=7
        )
        off_compare = bootstrap_off_comparison(
            oos_ret_raw=op["operation_return_raw"],
            oos_ret_off=op["operation_return"],
            spy_ret=op_spy,
            periods_per_year=data.periods_per_year,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            seed=42
        )
    
    return PipelineResult(
        freq=freq,
        periods_per_year=data.periods_per_year,
        data=data,
        signals=signals,
        
        oos_evaluation=oos_eval,
        operation=op,
        
        operation_sharpe=op_sharpe,
        operation_ir=op_ir,
        operation_mdd=op_mdd,
        
        confirmed_regime=confirmed,
        live_regime=live_snapshot,
        live_position=live_position,
        
        sector_analysis=sector_analysis,
        regime_perf=regime_perf,
        
        metrics={
            "cagr": cagr,
            "spy_cagr": spy_cagr,
            "ann_vol": ann_vol,
            "spy_vol": spy_vol,
            "hit_ratio": hit_ratio,
            "spy_hit": spy_hit,
        },
        
        bootstrap=bootstrap_res,
        permutation=perm_res,
        off_compare=off_compare,
        
        ui_params={
            "freq": freq,
            "pct_low_grid": pct_low_grid,
            "pct_high_grid": pct_high_grid,
            "enable_off": enable_off,
            "off_lookback": off_lookback,
            "off_mdd_threshold": off_mdd_threshold,
            "off_cooldown": off_cooldown,
            "run_bootstrap": run_bootstrap,
            "bootstrap_months": bootstrap_months,
            "bootstrap_override": bootstrap_override,
            "block_size_effective": block_size,
            "n_bootstrap": n_bootstrap,
            "n_perm": n_perm,
        }
    )


# ============================================================
# 11. SHINY UI (CONSUMER ONLY - FULL VERSION)
# ============================================================

LOW_GRID_CHOICES = list(range(10, 41, 5))
HIGH_GRID_CHOICES = list(range(60, 91, 5))

from shiny import App, ui, render, reactive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            body { font-family: Arial, sans-serif; }
            .metric-box { background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .warning-box { background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }
            .bootstrap-box { background-color: #e3f2fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }
            .data-table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            .data-table th, .data-table td { padding: 10px; text-align: left; border: 1px solid #ddd; }
            .data-table th { background-color: #4CAF50; color: white; }
            .pos { color: #4CAF50; font-weight: bold; }
            .neg { color: #f44336; font-weight: bold; }
            .sidebar-info { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 15px; }
            .loading { text-align: center; padding: 50px; color: #666; }
            .small-note { color:#666; font-size:12px; margin-top:6px; }
            .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
        """)
    ),

    ui.panel_title("Tech-Overlay Strategy Dashboard (Refactored Architecture)"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.div(
                {"class": "sidebar-info"},
                ui.h4("Strategy Info"),
                ui.p(f"p fixed: {P_FIXED:.2f}"),
                ui.p("Rebalance: W / 2W / M"),
            ),

            ui.hr(),

            ui.h5("Rebalance Frequency"),
            ui.input_radio_buttons(
                "freq",
                "Choose Frequency",
                choices={"W": "Weekly (W-FRI)", "2W": "Bi-Weekly (2W)", "M": "Monthly"},
                selected=REBALANCE_FREQ_DEFAULT,
                inline=False
            ),

            ui.hr(),

            ui.h5("Percentile Grid"),
            ui.input_checkbox_group(
                "low_grid",
                "Low Grid (10~40, step 5)",
                choices=[str(x) for x in LOW_GRID_CHOICES],
                selected=[str(x) for x in PERCENTILE_LOW_GRID_DEFAULT if x in LOW_GRID_CHOICES],
                inline=False
            ),
            ui.input_checkbox_group(
                "high_grid",
                "High Grid (60~90, step 5)",
                choices=[str(x) for x in HIGH_GRID_CHOICES],
                selected=[str(x) for x in PERCENTILE_HIGH_GRID_DEFAULT if x in HIGH_GRID_CHOICES],
                inline=False
            ),

            ui.hr(),

            ui.h5("OFF Switch"),
            ui.input_checkbox("enable_off", "Enable OFF Switch", value=ENABLE_OFF_SWITCH_DEFAULT),
            ui.input_numeric("off_lookback", "OFF_LOOKBACK", value=OFF_LOOKBACK_DEFAULT, min=1, max=52, step=1),
            ui.input_numeric("off_mdd", "OFF_MDD_THRESHOLD", value=OFF_MDD_THRESHOLD_DEFAULT, step=0.01),
            ui.input_numeric("off_cooldown", "OFF_COOLDOWN", value=OFF_COOLDOWN_DEFAULT, min=0, max=52, step=1),

            ui.hr(),

            ui.h5("Bootstrap Validation"),
            ui.input_checkbox("run_bootstrap", "Run Bootstrap + Permutation", value=True),
            ui.input_numeric("bootstrap_months", "Block Months", value=BOOTSTRAP_BLOCK_MONTHS_DEFAULT, min=1, max=12, step=1),
            ui.input_numeric("bootstrap_override", "Block Size Override", value=BOOTSTRAP_BLOCK_SIZE_OVERRIDE_DEFAULT, min=0, max=999, step=1),
            ui.input_numeric("n_bootstrap", "N_BOOTSTRAP", value=N_BOOTSTRAP_DEFAULT, min=100, max=5000, step=100),
            ui.input_numeric("n_perm", "N_PERM", value=N_PERM_DEFAULT, min=200, max=10000, step=200),

            ui.hr(),

            ui.input_action_button("run", "Run Backtest", class_="btn-primary btn-lg"),
            ui.br(), ui.br(),
            ui.hr(),
            ui.output_ui("sidebar_status"),
            width=340
        ),
        ui.output_ui("main_content")
    )
)

def server(input, output, session):

    state = reactive.Value(None)

    def _parse_grid(vals, default):
        if vals is None or len(vals) == 0:
            return default
        out = []
        for v in vals:
            try:
                out.append(int(v))
            except Exception:
                pass
        out = sorted(list(set(out)))
        return out if out else default

    def _parse_override(x):
        if x is None:
            return None
        try:
            v = int(x)
        except Exception:
            return None
        return None if v <= 0 else v

    @reactive.effect
    @reactive.event(input.run)
    def _():
        freq = str(input.freq())
        low_grid = _parse_grid(input.low_grid(), PERCENTILE_LOW_GRID_DEFAULT)
        high_grid = _parse_grid(input.high_grid(), PERCENTILE_HIGH_GRID_DEFAULT)

        enable_off = bool(input.enable_off())
        off_lookback = int(input.off_lookback())
        off_mdd = float(input.off_mdd())
        off_cooldown = int(input.off_cooldown())

        run_bootstrap = bool(input.run_bootstrap())
        bootstrap_months = int(input.bootstrap_months())
        override = _parse_override(input.bootstrap_override())
        n_bootstrap = int(input.n_bootstrap())
        n_perm = int(input.n_perm())

        with ui.Progress(min=0, max=100) as p:
            p.set(message="Loading data...", detail="Please wait")
            p.set(20)

            p.set(message="Running OOS + Operation...", detail=f"freq={freq}")
            p.set(55)

            if run_bootstrap:
                _ = infer_block_size(freq, bootstrap_months, override)
                p.set(message="Running validation...", detail=f"bootstrap={n_bootstrap}")
                p.set(80)

            res = run_pipeline(
                freq=freq,
                pct_low_grid=low_grid,
                pct_high_grid=high_grid,
                enable_off=enable_off,
                off_lookback=off_lookback,
                off_mdd_threshold=off_mdd,
                off_cooldown=off_cooldown,
                run_bootstrap=run_bootstrap,
                bootstrap_months=bootstrap_months,
                bootstrap_override=override,
                n_bootstrap=n_bootstrap,
                n_perm=n_perm,
            )
            p.set(100)
            state.set(res)

    @output
    @render.ui
    def sidebar_status():
        res = state.get()
        if res is None:
            return ui.div({"class": "sidebar-info"}, ui.p("Click 'Run Backtest' to start"))

        bs = res.bootstrap
        perm = res.permutation

        bs_text = ui.p("Bootstrap: OFF")
        if res.ui_params.get("run_bootstrap") and bs and bs.get("meta"):
            meta = bs["meta"]
            bs_text = ui.div(ui.p(f"Bootstrap: ON (n={meta['n_bootstrap']}, block={meta['block_size']})"))

        pval_text = ui.p("")
        if perm:
            pval_text = ui.p(f"IR p-value: {perm.get('p_value', np.nan):.4f}")

        confirmed = res.confirmed_regime
        live = res.live_regime
        live_obs = live.get("last_obs", pd.NaT)
        live_obs_str = live_obs.strftime("%Y-%m-%d") if pd.notna(live_obs) else "N/A"

        return ui.div(
            {"class": "sidebar-info"},
            ui.h5("Quick Stats (Operation)"),
            ui.p(f"Freq: {res.freq} | PPY: {res.periods_per_year}"),
            ui.p(f"Sharpe: {res.operation_sharpe:.3f}"),
            ui.p(f"IR: {res.operation_ir:.3f}"),
            ui.p(f"MDD: {res.operation_mdd*100:.2f}%"),
            ui.hr(),
            ui.h5("Regime Monitor"),
            ui.p(f"Confirmed @ {confirmed['date'].strftime('%Y-%m-%d')}: {confirmed['regime']} ({confirmed['percentile']:.1f}%)"),
            ui.p(f"Live @ {live['date'].strftime('%Y-%m-%d')}: {live['regime']} ({live['percentile']:.1f}%)"),
            ui.p(f"Last daily obs: {live_obs_str}"),
            ui.hr(),
            ui.h5("Validation"),
            bs_text,
            pval_text,
        )

    @output
    @render.ui
    def main_content():
        res = state.get()

        if res is None:
            return ui.div(
                {"class": "loading"},
                ui.h3("Dashboard"),
                ui.p("Click 'Run Backtest' in the sidebar to begin."),
            )

        op = res.operation
        confirmed = res.confirmed_regime
        live = res.live_regime
        live_pos = res.live_position
        op_ret = op["operation_return"]

        return ui.navset_tab(
            ui.nav_panel("Overview",
                ui.div(
                    {"class": "metric-box"},
                    ui.h3("Live Position (last completed bin)"),
                    ui.output_ui("live_position_ui")
                ),
                ui.div(
                    {"class": "warning-box"},
                    ui.strong("Note: "),
                    ui.span(
                        f"Operation Returns use closed bins only (ends at {op_ret.index[-1].strftime('%Y-%m-%d')}). "
                        f"Live regime/position is also based on the last completed bin."
                    )
                ),
                ui.h3("Regime Monitor"),
                ui.HTML(f"""
                    <p><strong>Confirmed Regime (parameter basis):</strong>
                    {confirmed['date'].strftime('%Y-%m-%d')} /
                    {confirmed['regime']} /
                    Percentile {confirmed['percentile']:.1f}% /
                    Thresholds Low {confirmed['pct_low_used']:.0f}%, High {confirmed['pct_high_used']:.0f}%</p>

                    <p><strong>Live Regime (last completed bin):</strong>
                    {live['date'].strftime('%Y-%m-%d')} /
                    {live['regime']} /
                    Percentile {live['percentile']:.1f}%</p>
                """),

                ui.h3("Operation Performance Summary"),
                ui.p(ui.em(f"Performance: {op_ret.index[0].strftime('%Y-%m-%d')} to {op_ret.index[-1].strftime('%Y-%m-%d')}")),
                ui.output_ui("performance_table"),
                ui.output_ui("bootstrap_summary_ui"),
            ),

            ui.nav_panel("Charts",
                ui.h3("Cumulative Return (log scale) - Operation"),
                ui.output_plot("cumulative_plot", height="520px"),
                ui.br(),
                ui.h3("Sector Selection Analysis (Operation)"),
                ui.output_plot("sector_plot", height="520px"),
            ),

            ui.nav_panel("Regime Analysis",
                ui.h3("Performance by Tech Regime (Operation)"),
                ui.output_ui("regime_table"),
                ui.br(),
                ui.h3("Sector Selection Frequency (Operation)"),
                ui.output_ui("sector_freq_text"),
            ),

            ui.nav_panel("Parameters",
                ui.h3("Parameter History (OOS evaluation)"),
                ui.output_ui("param_table"),
            ),

            ui.nav_panel("Bootstrap",
                ui.h3("Bootstrap CI + Permutation Test (Operation)"),
                ui.output_ui("bootstrap_details_ui"),
                ui.br(),
                ui.output_plot("bootstrap_dist_plot", height="520px"),
                ui.br(),
                ui.output_plot("perm_plot", height="420px"),
                ui.br(),
                ui.h3("OFF Switch Validation"),
                ui.output_ui("off_switch_validation_ui"),
                ui.br(),
                ui.output_plot("off_compare_sharpe_plot", height="420px"),
                ui.br(),
                ui.output_plot("off_compare_mdd_plot", height="420px"),
            ),
        )

    @output
    @render.ui
    def live_position_ui():
        res = state.get()
        if res is None:
            return None

        live_pos = res.live_position
        obs = live_pos.get("last_obs", pd.NaT)
        obs_str = obs.strftime("%Y-%m-%d") if pd.notna(obs) else "N/A"

        weights_rows = "".join([
            f"<tr><td>{s} ({SECTOR_ETFS.get(s, s)})</td><td>{w*100:.2f}%</td></tr>"
            for s, w in sorted(live_pos["weights"].items(), key=lambda x: x[1], reverse=True)
        ])

        return ui.HTML(f"""
            <p><strong>As of (bin end label):</strong> {live_pos['date'].strftime('%Y-%m-%d (%A)')}</p>
            <p><strong>Last daily obs inside bin:</strong> {obs_str}</p>
            <p><strong>Tech Percentile:</strong> <span style='font-size:24px; color:#4CAF50;'>{live_pos['percentile']:.1f}%</span></p>
            <p><strong>Regime:</strong> <span style='font-size:18px; font-weight:bold;'>{live_pos['regime']}</span></p>
            <p><strong>Thresholds (active params):</strong> Low {live_pos['pct_low_used']:.0f}%, High {live_pos['pct_high_used']:.0f}%</p>
            <table class='data-table'>
                <tr><th>Sector</th><th>Weight</th></tr>
                {weights_rows}
            </table>
        """)

    @output
    @render.ui
    def performance_table():
        res = state.get()
        if res is None:
            return None

        metrics = res.metrics
        op = res.operation

        spy_sharpe = calc_sharpe(op["operation_spy_return"], periods_per_year=res.periods_per_year)
        spy_mdd = calc_mdd(op["operation_spy_cumulative"])

        cagr = metrics["cagr"]
        spy_cagr = metrics["spy_cagr"]
        ann_vol = metrics["ann_vol"]
        spy_vol = metrics["spy_vol"]
        hit_ratio = metrics["hit_ratio"]
        spy_hit = metrics["spy_hit"]

        def color_class(val1, val2):
            if pd.notna(val1) and pd.notna(val2):
                return "pos" if val1 > val2 else "neg"
            return ""

        return ui.HTML(f"""
            <p class="small-note">freq=<span class="mono">{res.freq}</span> / periods_per_year=<span class="mono">{res.periods_per_year}</span></p>
            <table class='data-table'>
                <tr><th>Metric</th><th>Strategy</th><th>SPY</th><th>Diff</th></tr>
                <tr>
                    <td>CAGR</td>
                    <td>{fmt_pct(cagr)}</td>
                    <td>{fmt_pct(spy_cagr)}</td>
                    <td class='{color_class(cagr, spy_cagr)}'>{fmt_pct(cagr - spy_cagr)}</td>
                </tr>
                <tr>
                    <td>Ann. Vol</td>
                    <td>{fmt_pct(ann_vol)}</td>
                    <td>{fmt_pct(spy_vol)}</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Sharpe</td>
                    <td>{fmt_num(res.operation_sharpe, 3)}</td>
                    <td>{fmt_num(spy_sharpe, 3)}</td>
                    <td class='{color_class(res.operation_sharpe, spy_sharpe)}'>{fmt_num(res.operation_sharpe - spy_sharpe, 3)}</td>
                </tr>
                <tr>
                    <td>IR</td>
                    <td class='pos'>{fmt_num(res.operation_ir, 3)}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>MDD</td>
                    <td>{fmt_pct(res.operation_mdd)}</td>
                    <td>{fmt_pct(spy_mdd)}</td>
                    <td class='{color_class(res.operation_mdd, spy_mdd)}'>{fmt_pct(res.operation_mdd - spy_mdd)}</td>
                </tr>
                <tr>
                    <td>Hit Ratio</td>
                    <td>{fmt_pct(hit_ratio)}</td>
                    <td>{fmt_pct(spy_hit)}</td>
                    <td>-</td>
                </tr>
            </table>
        """)

    @output
    @render.ui
    def bootstrap_summary_ui():
        res = state.get()
        if not res:
            return None
        ui_params = res.ui_params
        if not ui_params.get("run_bootstrap"):
            return None

        bs = res.bootstrap
        perm = res.permutation
        if not bs or not bs.get("ci"):
            return ui.div({"class": "bootstrap-box"}, ui.p("Bootstrap 결과가 없습니다."))

        ci = bs["ci"]

        def ci_line(name, key, fmt="{:.3f}"):
            c = ci.get(key)
            if c is None:
                return f"<p><strong>{name}:</strong> N/A</p>"
            return f"<p><strong>{name}:</strong> {fmt.format(c['original'])}  [ {fmt.format(c['ci_lower'])} , {fmt.format(c['ci_upper'])} ]</p>"

        pval_html = ""
        if perm:
            p = perm.get('p_value', np.nan)
            color = "#4CAF50" if (np.isfinite(p) and p < 0.05) else "#f44336"
            pval_html = f"<p><strong>IR p-value:</strong> <span style='color:{color}; font-weight:bold;'>{p:.4f}</span></p>"

        return ui.HTML(f"""
            <div class='bootstrap-box'>
                <h4>Bootstrap CI (95%)</h4>
                {ci_line("Sharpe", "sharpe")}
                {ci_line("IR", "ir")}
                {ci_line("MDD", "mdd")}
                {ci_line("Hit Ratio", "hit_ratio", fmt="{:.4f}")}
                {pval_html}
            </div>
        """)

    @output
    @render.plot
    def off_compare_sharpe_plot():
        res = state.get()
        if not res:
            return None
        if not res.ui_params.get("run_bootstrap"):
            return None

        oc = res.off_compare
        if oc is None:
            return None

        sr_raw = np.array(oc["sharpe_raw"], dtype=float)
        sr_off = np.array(oc["sharpe_off"], dtype=float)
        sr_raw = sr_raw[np.isfinite(sr_raw)]
        sr_off = sr_off[np.isfinite(sr_off)]
        if len(sr_raw) == 0 or len(sr_off) == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(sr_raw, bins=50, alpha=0.55, edgecolor="black", label="OFF OFF (RAW)")
        ax.hist(sr_off, bins=50, alpha=0.55, edgecolor="black", label="OFF ON")
        ax.axvline(np.median(sr_raw), linestyle="--", linewidth=2, label=f"RAW median={np.median(sr_raw):.3f}")
        ax.axvline(np.median(sr_off), linestyle="--", linewidth=2, label=f"OFF median={np.median(sr_off):.3f}")
        ax.set_title("Bootstrap Sharpe Distribution: OFF OFF vs OFF ON", fontweight="bold")
        ax.set_xlabel("Sharpe")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        plt.tight_layout()
        plt.close(fig)
        return fig

    @output
    @render.plot
    def off_compare_mdd_plot():
        res = state.get()
        if not res:
            return None
        if not res.ui_params.get("run_bootstrap"):
            return None

        oc = res.off_compare
        if oc is None:
            return None

        mdd_raw = np.array(oc["mdd_raw"], dtype=float)
        mdd_off = np.array(oc["mdd_off"], dtype=float)
        mdd_raw = mdd_raw[np.isfinite(mdd_raw)]
        mdd_off = mdd_off[np.isfinite(mdd_off)]
        if len(mdd_raw) == 0 or len(mdd_off) == 0:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(mdd_raw, bins=50, alpha=0.55, edgecolor="black", label="OFF OFF (RAW)")
        ax.hist(mdd_off, bins=50, alpha=0.55, edgecolor="black", label="OFF ON")
        ax.axvline(np.percentile(mdd_raw, 95), linestyle="--", linewidth=2, label=f"RAW 95%={np.percentile(mdd_raw,95):.3f}")
        ax.axvline(np.percentile(mdd_off, 95), linestyle="--", linewidth=2, label=f"OFF 95%={np.percentile(mdd_off,95):.3f}")
        ax.set_title("Bootstrap MDD Distribution: OFF OFF vs OFF ON", fontweight="bold")
        ax.set_xlabel("MDD (more negative is worse)")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend()
        plt.tight_layout()
        plt.close(fig)
        return fig

    @output
    @render.plot
    def cumulative_plot():
        res = state.get()
        if res is None:
            return None

        cum = res.operation["operation_cumulative"].dropna()
        spy = res.operation["operation_spy_cumulative"].reindex(cum.index).dropna()
        cum = cum[cum > 0]
        spy = spy[spy > 0]
        if cum.empty or spy.empty:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(cum.index, cum.values, label="Strategy", linewidth=2)
        ax.plot(spy.index, spy.values, label="SPY", linewidth=2, alpha=0.7)
        ax.set_yscale("log")

        if res.operation["operation_off_mask"].any():
            off_pts = cum.loc[res.operation["operation_off_mask"]]
            if not off_pts.empty:
                ax.scatter(off_pts.index, off_pts.values, s=30, alpha=0.6, label="OFF Switch", zorder=5)

        ax.legend(fontsize=11)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Cumulative Return", fontsize=11)
        ax.set_title("Strategy vs Benchmark (Operation)", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()
        plt.close(fig)
        return fig

    @output
    @render.plot
    def sector_plot():
        res = state.get()
        if res is None:
            return None

        sa = res.sector_analysis

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        sel_freq = sa["selection_freq"].sort_values(ascending=False)
        colors = plt.cm.Set3(range(len(sel_freq)))
        ax1.barh(range(len(sel_freq)), sel_freq.values, color=colors)
        ax1.set_yticks(range(len(sel_freq)))
        ax1.set_yticklabels(sel_freq.index)
        ax1.set_xlabel("Selection Count", fontsize=11)
        ax1.set_title("Sector Selection Frequency (Operation)", fontsize=12, fontweight="bold")
        ax1.grid(axis="x", alpha=0.3, linestyle="--")

        regime_data = []
        regime_labels = []
        for r in ["Tech Depressed", "Tech Neutral", "Tech Overheated"]:
            if r in sa["regime_selection"]:
                regime_data.append(sa["regime_selection"][r])
                regime_labels.append(r.replace("Tech ", ""))

        if regime_data:
            regime_df = pd.DataFrame(regime_data, index=regime_labels).T
            im = ax2.imshow(regime_df.values, aspect="auto", cmap="YlOrRd")
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label("Count", fontsize=10)
            ax2.set_xticks(range(len(regime_df.columns)))
            ax2.set_xticklabels(regime_df.columns, fontsize=10)
            ax2.set_yticks(range(len(regime_df.index)))
            ax2.set_yticklabels(regime_df.index, fontsize=9)

            vmax = float(np.nanmax(regime_df.values)) if np.isfinite(regime_df.values).any() else 1.0
            for i in range(len(regime_df.index)):
                for j in range(len(regime_df.columns)):
                    val = regime_df.iloc[i, j]
                    text_color = "white" if val > vmax * 0.6 else "black"
                    ax2.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9, color=text_color, fontweight="bold")

            ax2.set_title("Selection by Regime (Operation)", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Regime", fontsize=11)
            ax2.set_ylabel("Sector", fontsize=11)

        plt.tight_layout()
        plt.close(fig)
        return fig

    @output
    @render.ui
    def regime_table():
        res = state.get()
        if res is None:
            return None

        regime_perf = res.regime_perf

        rows_html = ""
        for r, rp in regime_perf.items():
            rows_html += f"""
                <tr>
                    <td>{r}</td>
                    <td>{rp['count']}</td>
                    <td>{fmt_num(rp['strat_sharpe'], 3)}</td>
                    <td>{fmt_num(rp['spy_sharpe'], 3)}</td>
                    <td>{fmt_pct(rp['strat_cum'])}</td>
                    <td>{fmt_pct(rp['spy_cum'])}</td>
                </tr>
            """

        return ui.HTML(f"""
            <table class='data-table'>
                <tr>
                    <th>Regime</th>
                    <th>Periods</th>
                    <th>Strat Sharpe</th>
                    <th>SPY Sharpe</th>
                    <th>Strat Cum Return</th>
                    <th>SPY Cum Return</th>
                </tr>
                {rows_html}
            </table>
        """)

    @output
    @render.ui
    def sector_freq_text():
        res = state.get()
        if res is None:
            return None

        sa = res.sector_analysis
        freq = sa["selection_freq"].sort_values(ascending=False)

        rows = ""
        for sec, cnt in freq.items():
            rows += f"<tr><td>{sec}</td><td>{int(cnt)}</td></tr>"

        return ui.HTML(f"""
            <table class="data-table">
                <tr>
                    <th>Sector</th>
                    <th>Selection Count</th>
                </tr>
                {rows}
            </table>
        """)

    @output
    @render.ui
    def param_table():
        res = state.get()
        if res is None:
            return None

        ph = res.oos_evaluation["param_history"]

        rows_html = ""
        for _, row in ph.iterrows():
            rows_html += f"""
                <tr>
                    <td>{pd.Timestamp(row['OOS_start']).strftime('%Y-%m')} to {pd.Timestamp(row['OOS_end']).strftime('%Y-%m')}</td>
                    <td>{float(row['pct_low']):.0f}</td>
                    <td>{float(row['pct_high']):.0f}</td>
                    <td class='pos'>{fmt_num(row['ir'], 3)}</td>
                    <td>{fmt_num(row['sharpe'], 3)}</td>
                    <td>{fmt_pct(row['mdd'])}</td>
                </tr>
            """

        return ui.HTML(f"""
            <table class='data-table'>
                <tr>
                    <th>OOS Period</th>
                    <th>Low Threshold</th>
                    <th>High Threshold</th>
                    <th>IR (IS)</th>
                    <th>Sharpe (IS)</th>
                    <th>MDD (IS)</th>
                </tr>
                {rows_html}
            </table>
        """)

    @output
    @render.plot
    def bootstrap_dist_plot():
        res = state.get()
        if not res:
            return None
        if not res.ui_params.get("run_bootstrap"):
            return None

        bs = res.bootstrap
        if not bs or not bs.get("dist") or not bs.get("ci"):
            return None

        dist = bs["dist"]
        ci = bs["ci"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        items = [
            ("sharpe", "Sharpe"),
            ("ir", "IR"),
            ("mdd", "MDD"),
            ("hit_ratio", "Hit Ratio"),
        ]

        for ax, (k, title) in zip(axes, items):
            vals = np.array([v for v in dist[k] if np.isfinite(v)], dtype=float)
            if len(vals) == 0 or ci.get(k) is None:
                ax.set_title(f"{title} (N/A)")
                ax.axis("off")
                continue

            ax.hist(vals, bins=50, alpha=0.75, edgecolor="black")
            ax.axvline(ci[k]["original"], linestyle="--", linewidth=2)
            ax.axvline(ci[k]["ci_lower"], linestyle=":", linewidth=2)
            ax.axvline(ci[k]["ci_upper"], linestyle=":", linewidth=2)
            ax.set_title(title, fontweight="bold")
            ax.grid(alpha=0.25, linestyle="--")

        plt.tight_layout()
        plt.close(fig)
        return fig

    @output
    @render.plot
    def perm_plot():
        res = state.get()
        if not res:
            return None
        if not res.ui_params.get("run_bootstrap"):
            return None

        perm = res.permutation
        if not perm or perm.get("null_distribution") is None:
            return None

        null_irs = perm["null_distribution"]
        obs = perm.get("observed_ir", np.nan)
        p = perm.get("p_value", np.nan)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(null_irs, bins=60, alpha=0.75, edgecolor="black")
        ax.axvline(obs, linestyle="--", linewidth=2)
        ax.axvline(-obs, linestyle="--", linewidth=2)

        ax.set_title("Permutation Test (IR)", fontweight="bold")
        ax.set_xlabel("IR under H0")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.25, linestyle="--")

        ax.text(
            0.02, 0.96, f"p-value={p:.4f}", transform=ax.transAxes,
            va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )

        plt.tight_layout()
        plt.close(fig)
        return fig

app = App(app_ui, server)
