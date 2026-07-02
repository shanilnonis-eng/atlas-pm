"""
test_app_imports.py
--------------------
Smoke tests that verify every core module and every active page file
can be imported without raising an exception.

Design: each test only imports the module and checks that key symbols
are present. No live market data is fetched; Streamlit page execution
is NOT triggered (pages are modules, not run as scripts here).

If any import fails the test produces a clear message naming the module
so the developer can find and fix it quickly.
"""

from __future__ import annotations

import importlib
import sys
import os
import types
import pytest

# Make atlas-pm root importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Helper ───────────────────────────────────────────────────────────────────

def _import(module_path: str) -> types.ModuleType:
    """Import a dotted module path and return the module object."""
    return importlib.import_module(module_path)


# ─── Core library modules ──────────────────────────────────────────────────────

class TestCoreModuleImports:
    """Every calculation module must be importable and expose its public API."""

    def test_config_settings(self):
        mod = _import("config.settings")
        assert hasattr(mod, "UNIVERSE")
        assert hasattr(mod, "TRADING_DAYS_PER_YEAR")
        assert hasattr(mod, "VAR_CONFIDENCE")
        assert hasattr(mod, "MODEL_NAMES")
        assert hasattr(mod, "STRESS_SCENARIOS")

    def test_data_loader(self):
        mod = _import("data.loader")
        assert hasattr(mod, "load_prices")
        assert hasattr(mod, "compute_returns")
        assert hasattr(mod, "align_series")

    def test_analytics_returns(self):
        mod = _import("analytics.returns")
        for fn in [
            "total_return", "annualised_return", "annualised_volatility",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "max_drawdown", "drawdown_series", "cumulative_returns",
            "rolling_volatility", "rolling_sharpe", "monthly_returns_table",
            "portfolio_returns", "beta", "alpha", "information_ratio",
            "summary_statistics",
        ]:
            assert hasattr(mod, fn), f"analytics.returns missing: {fn}"

    def test_analytics_risk(self):
        mod = _import("analytics.risk")
        for fn in [
            "historical_var", "historical_cvar", "parametric_var",
            "var_summary", "marginal_risk_contribution",
            "component_risk_contribution", "correlation_matrix",
            "rolling_correlation", "run_stress_test", "var_backtesting",
        ]:
            assert hasattr(mod, fn), f"analytics.risk missing: {fn}"

    def test_analytics_factors(self):
        mod = _import("analytics.factors")
        for fn in [
            "load_ff3_factors", "synthetic_factors",
            "run_factor_regression", "regression_table",
            "rolling_factor_betas",
        ]:
            assert hasattr(mod, fn), f"analytics.factors missing: {fn}"

    def test_analytics_backtest(self):
        mod = _import("analytics.backtest")
        for name in [
            "run_walk_forward", "BacktestResult", "PeriodResult",
            "build_summary_table", "build_degradation_table",
        ]:
            assert hasattr(mod, name), f"analytics.backtest missing: {name}"

    def test_analytics_brinson_attribution(self):
        mod = _import("analytics.brinson_attribution")
        for fn in [
            "normalize_asset_name", "align_asset_names",
            "calculate_group_weights", "build_benchmark_weights",
            "calculate_group_returns", "calculate_brinson_attribution",
            "calculate_period_active_return", "calculate_cumulative_attribution",
            "validate_brinson_reconciliation", "calculate_ic_proxy",
            "generate_interpretation",
            "DEFAULT_CLASSIFICATION", "BENCHMARK_60_40_GROUP_WEIGHTS",
        ]:
            assert hasattr(mod, fn), f"analytics.brinson_attribution missing: {fn}"

    def test_analytics_turnover(self):
        mod = _import("analytics.turnover")
        for fn in ["compute_turnover", "rebalancing_cost", "simulate_rebalancing",
                   "turnover_comparison"]:
            assert hasattr(mod, fn), f"analytics.turnover missing: {fn}"

    def test_construction_optimiser(self):
        mod = _import("construction.optimiser")
        for fn in [
            "equal_weight", "minimum_variance", "maximum_sharpe",
            "risk_parity", "compute_cov_matrix", "ledoit_wolf_shrinkage",
            "efficient_frontier", "run_optimisation",
        ]:
            assert hasattr(mod, fn), f"construction.optimiser missing: {fn}"

    def test_construction_black_litterman(self):
        mod = _import("construction.black_litterman")
        assert hasattr(mod, "BlackLitterman")
        assert hasattr(mod, "View")
        bl_cls = mod.BlackLitterman
        for method in [
            "equilibrium_returns", "posterior_returns", "posterior_covariance",
            "optimal_weights", "add_view", "clear_views",
            "view_impact_table", "summary",
        ]:
            assert hasattr(bl_cls, method), f"BlackLitterman missing method: {method}"

    def test_ai_commentary(self):
        mod = _import("ai.commentary")
        for fn in [
            "generate_performance_commentary", "generate_risk_commentary",
            "generate_allocation_commentary", "generate_bull_base_bear",
            "generate_ic_report_narrative", "answer_question",
        ]:
            assert hasattr(mod, fn), f"ai.commentary missing: {fn}"

    def test_reporting_pdf_export(self):
        mod = _import("reporting.pdf_export")
        assert mod is not None   # just verify it imports without crashing


# ─── Public symbol type-checks ────────────────────────────────────────────────

class TestSymbolTypes:
    """Spot-check that key symbols are the expected type (callable, dict, etc.)."""

    def test_universe_is_dict(self):
        from config.settings import UNIVERSE
        assert isinstance(UNIVERSE, dict)
        assert len(UNIVERSE) > 0

    def test_trading_days_is_integer(self):
        from config.settings import TRADING_DAYS_PER_YEAR
        assert isinstance(TRADING_DAYS_PER_YEAR, int)
        assert TRADING_DAYS_PER_YEAR == 252

    def test_model_names_is_list_of_four(self):
        from config.settings import MODEL_NAMES
        assert isinstance(MODEL_NAMES, list)
        assert len(MODEL_NAMES) == 4
        assert "Equal Weight"      in MODEL_NAMES
        assert "Minimum Variance"  in MODEL_NAMES
        assert "Maximum Sharpe"    in MODEL_NAMES
        assert "Risk Parity"       in MODEL_NAMES

    def test_stress_scenarios_has_five_entries(self):
        from config.settings import STRESS_SCENARIOS
        assert isinstance(STRESS_SCENARIOS, dict)
        assert len(STRESS_SCENARIOS) == 5

    def test_var_confidence_in_range(self):
        from config.settings import VAR_CONFIDENCE
        assert 0 < VAR_CONFIDENCE < 1
        assert VAR_CONFIDENCE == 0.95

    def test_asset_short_names_covers_all_universe(self):
        from config.settings import UNIVERSE, ASSET_SHORT_NAMES
        for label in UNIVERSE:
            assert label in ASSET_SHORT_NAMES, (
                f"ASSET_SHORT_NAMES missing entry for universe asset: {label!r}"
            )

    def test_default_classification_covers_all_universe(self):
        from config.settings import UNIVERSE
        from analytics.brinson_attribution import DEFAULT_CLASSIFICATION
        for label in UNIVERSE:
            assert label in DEFAULT_CLASSIFICATION, (
                f"DEFAULT_CLASSIFICATION missing universe asset: {label!r}"
            )


# ─── Callable signatures (basic) ──────────────────────────────────────────────

class TestCallableSignatures:
    """Verify that key functions accept expected positional arguments without crashing."""

    def test_total_return_callable(self):
        import pandas as pd
        import numpy as np
        from analytics.returns import total_return
        r = pd.Series([0.01, -0.005, 0.02])
        result = total_return(r)
        assert isinstance(result, float)

    def test_historical_var_callable(self):
        import pandas as pd
        import numpy as np
        from analytics.risk import historical_var
        r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100))
        result = historical_var(r, 0.95, 1)
        assert isinstance(result, float)

    def test_equal_weight_callable(self):
        from construction.optimiser import equal_weight
        w = equal_weight(["A", "B", "C"])
        assert abs(w.sum() - 1.0) < 1e-10

    def test_run_stress_test_callable(self):
        from analytics.risk import run_stress_test
        weights = {"US Equities (S&P 500)": 0.5, "US Aggregate Bonds": 0.5}
        result = run_stress_test(weights)
        assert not result.empty

    def test_compute_returns_callable(self):
        import pandas as pd
        from data.loader import compute_returns
        prices = pd.DataFrame(
            {"A": [100.0, 105.0, 103.0]},
            index=pd.date_range("2020-01-01", periods=3),
        )
        simple, log = compute_returns(prices)
        assert len(simple) == 2
        assert len(log) == 2

    def test_run_optimisation_dispatcher(self):
        import numpy as np
        import pandas as pd
        from construction.optimiser import run_optimisation
        rng = np.random.default_rng(1)
        n, k = 200, 3
        rets = pd.DataFrame(
            rng.normal(0.001, 0.01, (n, k)),
            columns=[f"A{i}" for i in range(k)],
        )
        for model in ["Equal Weight", "Minimum Variance", "Maximum Sharpe", "Risk Parity"]:
            w = run_optimisation(model, rets, rf_annual=0.04)
            assert abs(w.sum() - 1.0) < 1e-5, f"{model}: weights don't sum to 1"
            assert (w.values >= -1e-8).all(), f"{model}: negative weight"

    def test_run_optimisation_invalid_model_raises(self):
        import numpy as np
        import pandas as pd
        from construction.optimiser import run_optimisation
        rets = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, 0.03]})
        with pytest.raises(ValueError, match="Unknown model"):
            run_optimisation("InvalidModel", rets)


# ─── Streamlit page file imports ──────────────────────────────────────────────

class TestPageFilesSyntax:
    """
    Verify that every active page file compiles without syntax errors.

    We use compile() rather than importlib.import_module() because importing
    Streamlit pages would trigger st.set_page_config() and run the full page
    code, which requires a live Streamlit server context.

    compile() catches: SyntaxError, IndentationError, encoding errors.
    It does NOT catch runtime import errors inside the page (those need live tests).
    """

    PAGES_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pages",
    )

    def _compile_page(self, filename: str):
        path = os.path.join(self.PAGES_DIR, filename)
        assert os.path.exists(path), f"Page file not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        try:
            compile(source, path, "exec")
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {filename}: {e}")

    def test_page_0_home(self):          self._compile_page("0_Home.py")
    def test_page_1_universe(self):      self._compile_page("1_Universe_and_Data.py")
    def test_page_2_construction(self):  self._compile_page("2_Portfolio_Construction.py")
    def test_page_3_performance(self):   self._compile_page("3_Performance_Analytics.py")
    def test_page_4_risk(self):          self._compile_page("4_Risk_Management.py")
    def test_page_5_ai(self):            self._compile_page("5_AI_Commentary.py")
    def test_page_6_ic_report(self):     self._compile_page("6_IC_Report.py")
    def test_page_7_bl(self):            self._compile_page("7_Black_Litterman.py")
    def test_page_8_factor(self):        self._compile_page("8_Factor_Attribution.py")
    def test_page_9_walkforward(self):   self._compile_page("9_Walkforward_Backtest.py")
    def test_page_10_brinson(self):      self._compile_page("10_Brinson_Attribution.py")
