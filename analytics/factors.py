"""
Factor attribution module for Atlas PM.

Decomposes portfolio returns into systematic factor exposures using the
Fama-French 3-factor model (and optionally 5-factor).

Fama-French 3 Factors:
  Mkt-RF : Excess market return (equity risk premium)
  SMB    : Small Minus Big (size premium — small caps outperform large caps)
  HML    : High Minus Low (value premium — value outperforms growth)

The regression:
    R_p - R_f = α + β_mkt(Mkt-RF) + β_smb(SMB) + β_hml(HML) + ε

Interpretation:
  α       : Jensen's alpha — return not explained by factor exposures
  β_mkt   : market beta — sensitivity to broad equity market
  β_smb   : size tilt — positive = small-cap tilt, negative = large-cap
  β_hml   : value tilt — positive = value tilt, negative = growth

Data source: Ken French Data Library
  (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
  Accessed via pandas_datareader.

Reference: Fama, E.F. & French, K.R. (1993) 'Common risk factors in the
returns on stocks and bonds', Journal of Financial Economics, 33(1), 3-56.
"""

from __future__ import annotations

import io
import zipfile
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    import pandas_datareader.data as web
    _DATAREADER_AVAILABLE = True
except ImportError:
    _DATAREADER_AVAILABLE = False

try:
    import urllib.request
    _URLLIB_AVAILABLE = True
except ImportError:
    _URLLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Factor data loading
# ---------------------------------------------------------------------------

def _download_ff3_direct(frequency: str = "monthly") -> pd.DataFrame | None:
    """
    Download Fama-French 3-factor data directly from the Ken French website.
    More reliable than pandas_datareader which can time out or change API.
    """
    import urllib.request
    import zipfile
    import io

    if frequency == "daily":
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    else:
        url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            zip_data = response.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
            csv_name = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
            with z.open(csv_name) as f:
                raw_text = f.read().decode("utf-8", errors="ignore")

        # Ken French CSVs have a header description block before the data.
        # Find the line where the actual data starts (first line with 4 numeric columns).
        lines = raw_text.splitlines()
        data_start = 0
        for i, line in enumerate(lines):
            parts = line.strip().split(",")
            if len(parts) >= 4 and parts[0].strip().lstrip("-").isdigit():
                data_start = i
                break

        # find where the data ends (blank line or non-numeric after data starts)
        data_end = len(lines)
        for i in range(data_start + 1, len(lines)):
            parts = lines[i].strip().split(",")
            if not lines[i].strip() or not parts[0].strip().lstrip("-").isdigit():
                data_end = i
                break

        data_lines = lines[data_start:data_end]
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            header=None,
            names=["Date", "Mkt-RF", "SMB", "HML", "RF"],
            index_col=0,
        )
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        df = df / 100  # convert from % to decimal

        # parse date index
        if frequency == "daily":
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
        else:
            df.index = pd.to_datetime(df.index.astype(str), format="%Y%m", errors="coerce")
            df.index = df.index + pd.offsets.MonthEnd(0)

        df = df.dropna()
        return df

    except Exception:
        return None


def load_ff3_factors(
    start: str,
    end: str,
    frequency: str = "monthly",
) -> pd.DataFrame | None:
    """
    Load Fama-French 3-factor data.

    Tries direct download from Ken French website first (most reliable),
    then falls back to pandas_datareader.

    Returns
    -------
    DataFrame with columns: Mkt-RF, SMB, HML, RF (all as fractions, not %)
    or None if data cannot be fetched.
    """
    # Method 1: direct download (most reliable)
    df = _download_ff3_direct(frequency)

    # Method 2: pandas_datareader fallback
    if df is None and _DATAREADER_AVAILABLE:
        try:
            dataset = "F-F_Research_Data_Factors_daily" if frequency == "daily" else "F-F_Research_Data_Factors"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = web.DataReader(dataset, "famafrench", start=start, end=end)
            df_raw = raw[0].copy() if isinstance(raw, tuple) else raw.copy()
            df_raw = df_raw / 100
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            rename_map = {}
            for col in df_raw.columns:
                if "Mkt" in col: rename_map[col] = "Mkt-RF"
                elif "SMB" in col: rename_map[col] = "SMB"
                elif "HML" in col: rename_map[col] = "HML"
                elif col == "RF": rename_map[col] = "RF"
            df_raw = df_raw.rename(columns=rename_map)
            if all(c in df_raw.columns for c in ["Mkt-RF", "SMB", "HML", "RF"]):
                df = df_raw[["Mkt-RF", "SMB", "HML", "RF"]].dropna()
        except Exception:
            pass

    if df is None:
        return None

    # filter to requested date range
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    return df if not df.empty else None


def synthetic_factors(
    market_returns: pd.Series,
    rf_returns: pd.Series,
    frequency: str = "monthly",
) -> pd.DataFrame:
    """
    Fallback: construct simplified factor proxies from available data
    when the Ken French library is unavailable.

    This is a rough approximation — SMB and HML are set to zero.
    Clearly documented in the UI as a degraded fallback.
    """
    mkt_rf = market_returns.subtract(rf_returns, fill_value=0)

    if frequency == "monthly":
        mkt_rf = (1 + mkt_rf).resample("ME").prod() - 1

    df = pd.DataFrame({
        "Mkt-RF": mkt_rf,
        "SMB":    0.0,
        "HML":    0.0,
        "RF":     rf_returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
                  if frequency == "monthly" else rf_returns,
    }).dropna()
    return df


# ---------------------------------------------------------------------------
# Factor regression
# ---------------------------------------------------------------------------

def run_factor_regression(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    frequency: str = "monthly",
) -> dict:
    """
    OLS regression of portfolio excess returns on Fama-French factors.

    Parameters
    ----------
    portfolio_returns : daily simple returns Series
    factors           : DataFrame with columns Mkt-RF, SMB, HML, RF
                        (must match the frequency parameter)
    frequency         : 'daily' or 'monthly'

    Returns
    -------
    dict with regression coefficients, t-stats, R², adjusted R², and residuals
    """
    # resample portfolio to monthly if needed
    if frequency == "monthly":
        port_monthly = (1 + portfolio_returns).resample("ME").prod() - 1
        port_monthly.index = port_monthly.index.to_period("M").to_timestamp("M")
        factors_aligned = factors.copy()
        factors_aligned.index = factors_aligned.index.to_period("M").to_timestamp("M")
    else:
        port_monthly = portfolio_returns
        factors_aligned = factors

    # align
    combined = pd.concat([port_monthly, factors_aligned], axis=1, join="inner").dropna()
    combined.columns = ["Port"] + list(factors_aligned.columns)

    # excess portfolio return
    y = combined["Port"] - combined["RF"]

    # drop zero-variance factor columns (e.g. synthetic fallback where SMB=HML=0)
    factor_cols = [f for f in ["Mkt-RF", "SMB", "HML"] if combined[f].std() > 1e-12]
    X = combined[factor_cols]

    # OLS via numpy (avoids statsmodels dependency)
    X_mat = np.column_stack([np.ones(len(X)), X.values])
    b, residuals, rank, sv = np.linalg.lstsq(X_mat, y.values, rcond=None)

    alpha_monthly = b[0]
    beta_mkt  = b[1] if "Mkt-RF" in factor_cols else 0.0
    beta_smb  = b[factor_cols.index("SMB") + 1] if "SMB" in factor_cols else 0.0
    beta_hml  = b[factor_cols.index("HML") + 1] if "HML" in factor_cols else 0.0

    # annualise alpha
    periods_per_year = 12 if frequency == "monthly" else 252
    alpha_annual = (1 + alpha_monthly) ** periods_per_year - 1

    # t-statistics and R²
    y_pred = X_mat @ b
    resid  = y.values - y_pred
    k = 1 + len(factor_cols)   # intercept + active factors
    n = len(y)
    sigma2 = np.sum(resid ** 2) / max(n - k, 1)
    try:
        cov_b = sigma2 * np.linalg.inv(X_mat.T @ X_mat)
        se_b  = np.sqrt(np.diag(cov_b))
    except np.linalg.LinAlgError:
        se_b = np.ones(len(b)) * float("nan")

    t_stats  = b / se_b
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=max(n - k, 1)))

    def _get_ts(factor: str) -> tuple[float, float]:
        """Return (t-stat, p-value) for a factor, or (nan, nan) if not in model."""
        if factor not in factor_cols:
            return float("nan"), float("nan")
        idx = factor_cols.index(factor) + 1  # +1 for intercept
        return float(t_stats[idx]), float(p_values[idx])

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y.values - y.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    r2_adj = 1 - (1 - r2) * (n - 1) / max(n - k, 1)

    # factor contribution to return (average factor × beta)
    avg_factors = X.mean()
    contributions = {
        "Alpha (monthly)": alpha_monthly,
        "Alpha (annual)":  alpha_annual,
        "Market":          float(avg_factors.get("Mkt-RF", 0) * beta_mkt),
        "Size (SMB)":      float(avg_factors.get("SMB", 0)    * beta_smb),
        "Value (HML)":     float(avg_factors.get("HML", 0)    * beta_hml),
    }

    t_mkt, p_mkt = _get_ts("Mkt-RF")
    t_smb, p_smb = _get_ts("SMB")
    t_hml, p_hml = _get_ts("HML")

    return {
        "alpha_monthly":    alpha_monthly,
        "alpha_annual":     alpha_annual,
        "beta_mkt":         beta_mkt,
        "beta_smb":         beta_smb,
        "beta_hml":         beta_hml,
        "t_alpha":          float(t_stats[0]),
        "t_mkt":            t_mkt,
        "t_smb":            t_smb,
        "t_hml":            t_hml,
        "p_alpha":          float(p_values[0]),
        "p_mkt":            p_mkt,
        "p_smb":            p_smb,
        "p_hml":            p_hml,
        "r_squared":        r2,
        "r_squared_adj":    r2_adj,
        "n_observations":   n,
        "residuals":        pd.Series(resid, index=combined.index, name="Residuals"),
        "fitted":           pd.Series(y_pred, index=combined.index, name="Fitted"),
        "actual_excess":    y,
        "contributions":    contributions,
        "frequency":        frequency,
    }


def regression_table(result: dict) -> pd.DataFrame:
    """Format regression result as a display table."""
    stars = lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    rows = [
        {
            "Factor":      "Alpha (intercept)",
            "Coefficient": f"{result['alpha_monthly']:.5f}",
            "t-stat":      f"{result['t_alpha']:.2f}",
            "p-value":     f"{result['p_alpha']:.4f}{stars(result['p_alpha'])}",
            "Significance":"***" if result["p_alpha"] < 0.01 else
                           "**" if result["p_alpha"] < 0.05 else
                           "*"  if result["p_alpha"] < 0.10 else "",
        },
        {
            "Factor":      "Market (Mkt-RF)",
            "Coefficient": f"{result['beta_mkt']:.4f}",
            "t-stat":      f"{result['t_mkt']:.2f}",
            "p-value":     f"{result['p_mkt']:.4f}{stars(result['p_mkt'])}",
            "Significance": stars(result["p_mkt"]),
        },
        {
            "Factor":      "Size (SMB)",
            "Coefficient": f"{result['beta_smb']:.4f}",
            "t-stat":      f"{result['t_smb']:.2f}",
            "p-value":     f"{result['p_smb']:.4f}{stars(result['p_smb'])}",
            "Significance": stars(result["p_smb"]),
        },
        {
            "Factor":      "Value (HML)",
            "Coefficient": f"{result['beta_hml']:.4f}",
            "t-stat":      f"{result['t_hml']:.2f}",
            "p-value":     f"{result['p_hml']:.4f}{stars(result['p_hml'])}",
            "Significance": stars(result["p_hml"]),
        },
    ]
    df = pd.DataFrame(rows)
    df.attrs["r_squared"]     = result["r_squared"]
    df.attrs["r_squared_adj"] = result["r_squared_adj"]
    df.attrs["n_obs"]         = result["n_observations"]
    return df


def rolling_factor_betas(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    window: int = 24,  # months
    frequency: str = "monthly",
) -> pd.DataFrame:
    """
    Compute rolling factor betas over a trailing window.

    Returns a DataFrame with columns: alpha, beta_mkt, beta_smb, beta_hml
    """
    if frequency == "monthly":
        port = (1 + portfolio_returns).resample("ME").prod() - 1
        port.index = port.index.to_period("M").to_timestamp("M")
        factors = factors.copy()
        factors.index = factors.index.to_period("M").to_timestamp("M")
    else:
        port = portfolio_returns

    combined = pd.concat([port, factors], axis=1, join="inner").dropna()
    combined.columns = ["Port"] + list(factors.columns)
    y_all = combined["Port"] - combined["RF"]
    X_all = combined[["Mkt-RF", "SMB", "HML"]]

    results = []
    for end_i in range(window, len(combined) + 1):
        start_i = end_i - window
        y_w = y_all.iloc[start_i:end_i]
        X_w = X_all.iloc[start_i:end_i]
        X_mat = np.column_stack([np.ones(len(X_w)), X_w.values])
        try:
            b, _, _, _ = np.linalg.lstsq(X_mat, y_w.values, rcond=None)
            results.append({
                "Date":      combined.index[end_i - 1],
                "Alpha":     b[0],
                "Mkt-RF β": b[1],
                "SMB β":    b[2],
                "HML β":    b[3],
            })
        except Exception:
            pass

    return pd.DataFrame(results).set_index("Date") if results else pd.DataFrame()
