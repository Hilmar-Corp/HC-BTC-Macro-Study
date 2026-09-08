from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from hc_macro_integration.paths import (
    FIGURES_DIR,
    MARKET_DIR,
    TABLES_DIR,
)
from hc_macro_integration.robustness.ablations import (
    FACTOR_SETS,
    ensure_sp500,
    exact_supf_scan,
    prepare_extended_data,
)


HC_NAVY = "#052D52"
HC_CHARCOAL = "#303A40"
HC_MID = "#7A858E"
HC_LIGHT = "#D8DDE1"
HC_PALE = "#F5F7F8"
HC_WHITE = "#FFFFFF"

PERIODS = [
    "pre_2020",
    "post_covid_pre_etf",
    "spot_etf_era",
]

PERIOD_LABELS = {
    "pre_2020": "Pré-2020",
    "post_covid_pre_etf": "2020–2023",
    "spot_etf_era": "Depuis 2024",
}

FACTOR_LABELS = {
    "ndx": "Nasdaq-100",
    "usd": "Dollar",
    "real_rate": "Taux réel",
    "credit": "Crédit",
    "ndx_log_return": "Nasdaq-100",
    "dollar_log_return": "Dollar",
    "real_rate_change": "Taux réel US 10 ans",
    "credit_spread_change": "Spread de crédit",
}

EVENT_LABELS = {
    "NDX_DOWN_10": "Nasdaq — queue basse",
    "NDX_UP_10": "Nasdaq — queue haute",
    "RISK_OFF_2OF3": "Risque défavorable",
    "RISK_ON_2OF3": "Risque favorable",
    "VIX_ORTHO_UP_10": "VIX orthogonal — queue haute",
    "CREDIT_WIDEN_10": "Crédit — élargissement",
}

HORIZON_LABELS = {
    "SAME_DAY": "Même jour",
    "FWD_1": "+1",
    "FWD_5": "+5",
    "FWD_20": "+20",
}

PUBLICATION_FIGURES = (
    "01_synchronisation_temporelle.png",
    "02_r2_ajuste_par_periode.png",
    "03_r2_roulant_252.png",
    "04_beta_nasdaq_par_periode.png",
    "05_betas_roulants_252.png",
    "06_shapley_r2_absolu.png",
    "07_nasdaq_vs_sp500.png",
    "08_profils_rupture.png",
    "09_evenements_meme_jour.png",
    "10_evenements_horizons.png",
)

PUBLICATION_DIR = FIGURES_DIR / "publication"


def _require_table(name: str) -> pd.DataFrame:
    path = TABLES_DIR / name

    if not path.exists():
        raise FileNotFoundError(
            "Table requise absente: "
            f"{path}. Exécuter d'abord `hc-macro all`."
        )

    return pd.read_csv(path)


def _style() -> Any:
    return {
        "figure.facecolor": HC_WHITE,
        "axes.facecolor": HC_WHITE,
        "axes.edgecolor": HC_CHARCOAL,
        "axes.labelcolor": HC_CHARCOAL,
        "axes.titlecolor": HC_CHARCOAL,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.color": HC_CHARCOAL,
        "ytick.color": HC_CHARCOAL,
        "text.color": HC_CHARCOAL,
        "font.size": 10,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": HC_LIGHT,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.55,
    }


def _save(fig: Figure, filename: str) -> Path:
    PUBLICATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = PUBLICATION_DIR / filename

    fig.savefig(
        path,
        dpi=260,
        bbox_inches="tight",
        facecolor=HC_WHITE,
    )

    plt.close(fig)
    return path


def _add_regime_markers(ax: Axes) -> None:
    for date in [
        pd.Timestamp("2020-03-09"),
        pd.Timestamp("2024-01-11"),
    ]:
        ax.axvline(
            cast(Any, date),
            linestyle="--",
            linewidth=0.9,
            color=HC_MID,
            alpha=0.8,
        )


def plot_01_synchronisation_temporelle() -> Path:
    with plt.rc_context(_style()):
        fig, ax = plt.subplots(
            figsize=(10.5, 3.6)
        )

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.1, 2.2)
        ax.axis("off")

        left = 0.14
        right = 0.86

        ax.plot(
            [left, right],
            [1.55, 1.55],
            linewidth=3.0,
            color=HC_CHARCOAL,
            solid_capstyle="round",
        )

        ax.plot(
            [left, right],
            [0.55, 0.55],
            linewidth=3.0,
            color=HC_NAVY,
            solid_capstyle="round",
        )

        for x in [left, right]:
            ax.plot(
                [x, x],
                [0.28, 1.82],
                linewidth=0.9,
                linestyle="--",
                color=HC_MID,
            )

        ax.scatter(
            [left, right],
            [1.55, 1.55],
            s=42,
            color=HC_CHARCOAL,
            zorder=3,
        )

        ax.scatter(
            [left, right],
            [0.55, 0.55],
            s=42,
            color=HC_NAVY,
            zorder=3,
        )

        ax.text(
            0.01,
            1.55,
            "Nasdaq",
            va="center",
            ha="left",
            fontweight="bold",
        )

        ax.text(
            0.01,
            0.55,
            "Bitcoin 24/7",
            va="center",
            ha="left",
            fontweight="bold",
            color=HC_NAVY,
        )

        ax.text(
            left,
            1.98,
            r"$\tau_{t-1}$",
            ha="center",
            va="center",
            fontsize=12,
        )

        ax.text(
            right,
            1.98,
            r"$\tau_t$",
            ha="center",
            va="center",
            fontsize=12,
        )

        ax.annotate(
            "même intervalle économique",
            xy=((left + right) / 2, 1.24),
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=HC_NAVY,
        )

        ax.annotate(
            "",
            xy=(right, 1.05),
            xytext=(left, 1.05),
            arrowprops={
                "arrowstyle": "<->",
                "linewidth": 1.2,
                "color": HC_NAVY,
            },
        )

        ax.set_title(
            "Synchronisation de Bitcoin sur la clôture effective du Nasdaq",
            pad=8,
        )

        fig.tight_layout()
        return _save(
            fig,
            PUBLICATION_FIGURES[0],
        )


def plot_02_r2_ajuste_par_periode() -> Path:
    df = _require_table(
        "period_metrics.csv"
    )

    current = (
        df.loc[df["variant"] == "primary"]
        .set_index("period")
        .loc[PERIODS]
    )

    values = (
        100.0
        * current["adj_r2"].to_numpy(
            dtype=float
        )
    )

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(
            figsize=(8.8, 5.1)
        )

        bars = ax.bar(
            [PERIOD_LABELS[p] for p in PERIODS],
            values,
            color=[HC_LIGHT, HC_CHARCOAL, HC_NAVY],
            width=0.58,
        )

        ax.axhline(
            0,
            linewidth=0.9,
            color=HC_CHARCOAL,
        )

        ax.set_ylabel("R² ajusté (%)")
        ax.set_title(
            "Le pouvoir explicatif devient matériel après 2020"
        )
        ax.grid(axis="y")

        for bar, value in zip(
            bars,
            values,
            strict=True,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value
                + (0.45 if value >= 0 else -0.45),
                f"{value:.1f} %",
                ha="center",
                va=(
                    "bottom"
                    if value >= 0
                    else "top"
                ),
                fontweight="bold",
            )

        fig.tight_layout()
        return _save(
            fig,
            PUBLICATION_FIGURES[1],
        )


def plot_03_r2_roulant_252() -> Path:
    df = _require_table(
        "rolling_macro_window_252.csv"
    )

    dates = pd.to_datetime(df["date"])
    values = 100.0 * df["adj_r2"].to_numpy(dtype=float)

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(
            figsize=(10.5, 5.1)
        )

        ax.plot(
            dates,
            values,
            linewidth=1.8,
            color=HC_NAVY,
        )

        ax.axhline(
            0,
            linewidth=0.9,
            color=HC_CHARCOAL,
        )

        _add_regime_markers(ax)

        ax.set_ylabel("R² ajusté (%)")
        ax.set_xlabel("Date")
        ax.set_title(
            "Pouvoir explicatif macrofinancier roulant — 252 séances"
        )
        ax.grid(axis="y")

        fig.tight_layout()
        return _save(
            fig,
            PUBLICATION_FIGURES[2],
        )


def plot_04_beta_nasdaq_par_periode() -> Path:
    df = _require_table(
        "interaction_period_betas.csv"
    )

    current = (
        df.loc[
            (df["variant"] == "primary")
            & (df["factor"] == "ndx")
        ]
        .set_index("period")
        .loc[PERIODS]
    )

    values = current["beta"].to_numpy(
        dtype=float
    )

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(
            figsize=(8.8, 5.1)
        )

        bars = ax.bar(
            [PERIOD_LABELS[p] for p in PERIODS],
            values,
            color=[HC_LIGHT, HC_CHARCOAL, HC_NAVY],
            width=0.58,
        )

        ax.axhline(
            0,
            linewidth=0.9,
            color=HC_CHARCOAL,
        )

        ax.set_ylabel("Beta Nasdaq-100")
        ax.set_title(
            "Sensibilité contemporaine de Bitcoin au Nasdaq-100"
        )
        ax.grid(axis="y")

        for bar, value in zip(
            bars,
            values,
            strict=True,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        fig.tight_layout()
        return _save(
            fig,
            PUBLICATION_FIGURES[3],
        )


def plot_05_betas_roulants_252() -> Path:
    df = _require_table(
        "rolling_macro_window_252.csv"
    )

    dates = pd.to_datetime(df["date"])

    factors = [
        "ndx_log_return",
        "dollar_log_return",
        "real_rate_change",
        "credit_spread_change",
    ]

    with plt.rc_context(_style()):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(11.0, 7.4),
            sharex=True,
        )

        for ax, factor in zip(
            axes.flat,
            factors,
            strict=True,
        ):
            beta = df[
                f"beta__{factor}"
            ].to_numpy(dtype=float)

            low = df[
                f"ci95_low__{factor}"
            ].to_numpy(dtype=float)

            high = df[
                f"ci95_high__{factor}"
            ].to_numpy(dtype=float)

            ax.fill_between(
                dates,
                low,
                high,
                color=HC_LIGHT,
                alpha=0.75,
                linewidth=0,
            )

            ax.plot(
                dates,
                beta,
                linewidth=1.45,
                color=HC_NAVY,
            )

            ax.axhline(
                0,
                linewidth=0.8,
                color=HC_CHARCOAL,
            )

            _add_regime_markers(ax)
            ax.set_title(
                FACTOR_LABELS[factor],
                fontsize=10.5,
            )
            ax.grid(axis="y")

        axes[0, 0].set_ylabel("Beta")
        axes[1, 0].set_ylabel("Beta")
        axes[1, 0].set_xlabel("Date")
        axes[1, 1].set_xlabel("Date")

        fig.suptitle(
            "Sensibilités roulantes — fenêtre de 252 séances\n"
            "Bandes ponctuelles à 95 % fondées sur les erreurs-types HAC",
            fontsize=12,
            fontweight="bold",
            color=HC_CHARCOAL,
            y=0.995,
        )

        fig.tight_layout(
            rect=(0, 0, 1, 0.94)
        )

        return _save(
            fig,
            PUBLICATION_FIGURES[4],
        )


def plot_06_shapley_r2_absolu() -> Path:
    df = _require_table(
        "shapley_r2_by_period.csv"
    )

    current = df.loc[
        df["variant"] == "primary"
    ].copy()

    factors = [
        "ndx",
        "usd",
        "real_rate",
        "credit",
    ]

    palette = [
        HC_NAVY,
        HC_CHARCOAL,
        HC_MID,
        HC_LIGHT,
    ]

    pivot = (
        current.pivot(
            index="period",
            columns="factor",
            values="shapley_r2",
        )
        .loc[PERIODS, factors]
        * 100.0
    )

    full_r2 = (
        current.groupby("period")[
            "full_r2"
        ]
        .first()
        .loc[PERIODS]
        .to_numpy(dtype=float)
        * 100.0
    )

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(
            figsize=(9.4, 5.7)
        )

        x = np.arange(len(PERIODS))
        bottom = np.zeros(len(PERIODS))

        for factor, color in zip(
            factors,
            palette,
            strict=True,
        ):
            values = pivot[
                factor
            ].to_numpy(dtype=float)

            ax.bar(
                x,
                values,
                bottom=bottom,
                width=0.58,
                color=color,
                label=FACTOR_LABELS[factor],
            )

            bottom = bottom + values

        ax.set_xticks(
            x,
            [PERIOD_LABELS[p] for p in PERIODS],
        )
        ax.set_ylabel("Contribution au R² (points de %)")
        ax.set_title(
            "Décomposition exacte de Shapley du pouvoir explicatif"
        )
        ax.grid(axis="y")
        ax.legend(
            ncol=2,
            loc="upper left",
        )

        for i, total in enumerate(full_r2):
            ax.text(
                i,
                total + 0.45,
                f"R² = {total:.1f} %",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.text(
            0.01,
            -0.16,
            "Pré-2020 : le R² total est proche de zéro ; les parts relatives "
            "doivent être interprétées avec prudence.",
            transform=ax.transAxes,
            fontsize=8.8,
            color=HC_MID,
        )

        fig.tight_layout()
        return _save(
            fig,
            PUBLICATION_FIGURES[5],
        )


def plot_07_nasdaq_vs_sp500() -> Path:
    df = _require_table(
        "phase5_period_factor_sets.csv"
    )

    current = df.loc[
        (df["variant"] == "primary")
        & df["factor_set"].isin(
            ["NDX_ONLY", "SPX_ONLY"]
        )
    ].copy()

    pivot = (
        current.pivot(
            index="period",
            columns="factor_set",
            values="adj_r2",
        )
        .loc[PERIODS]
        * 100.0
    )

    x = np.arange(len(PERIODS))
    width = 0.32

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(
            figsize=(9.4, 5.2)
        )

        ax.bar(
            x - width / 2,
            pivot["NDX_ONLY"].to_numpy(
                dtype=float
            ),
            width,
            color=HC_NAVY,
            label="Nasdaq-100",
        )

        ax.bar(
            x + width / 2,
            pivot["SPX_ONLY"].to_numpy(
                dtype=float
            ),
            width,
            color=HC_CHARCOAL,
            label="S&P 500",
        )

        ax.axhline(
            0,
            linewidth=0.9,
            color=HC_CHARCOAL,
        )

        ax.set_xticks(
            x,
            [PERIOD_LABELS[p] for p in PERIODS],
        )
        ax.set_ylabel("R² ajusté (%)")
        ax.set_title(
            "Le changement ne dépend pas d'une définition exclusivement Nasdaq"
        )
        ax.grid(axis="y")
        ax.legend()

        fig.tight_layout()
        return _save(
            fig,
            PUBLICATION_FIGURES[6],
        )


def _load_or_rebuild_break_scans() -> pd.DataFrame:
    cached = TABLES_DIR / "publication_break_scan_full.csv"

    if cached.exists():
        result = pd.read_csv(cached)
        result["date"] = pd.to_datetime(
            result["date"]
        )
        return result

    spx_path = MARKET_DIR / "sp500_daily.parquet"

    if not spx_path.exists():
        raise FileNotFoundError(
            "Données S&P 500 absentes. Exécuter d'abord `hc-macro robustness` "
            "ou `hc-macro all`; la génération des figures de publication "
            "ne télécharge pas silencieusement de nouvelles données."
        )

    spx = ensure_sp500()
    frames: list[pd.DataFrame] = []

    for variant in ["primary", "strict"]:
        data = prepare_extended_data(
            variant,
            spx,
        )

        for factor_set in [
            "NDX_ONLY",
            "FULL_NDX",
            "SPX_ONLY",
            "FULL_SPX",
        ]:
            scan, _ = exact_supf_scan(
                data=data,
                factors=FACTOR_SETS[factor_set],
            )

            scan = scan.copy()
            scan.insert(
                0,
                "factor_set",
                factor_set,
            )
            scan.insert(
                0,
                "variant",
                variant,
            )
            frames.append(scan)

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    result.to_csv(
        cached,
        index=False,
    )

    result["date"] = pd.to_datetime(
        result["date"]
    )

    return result


def plot_08_profils_rupture() -> Path:
    df = _load_or_rebuild_break_scans()

    current = df.loc[
        df["variant"] == "primary"
    ].copy()

    factor_sets = [
        "NDX_ONLY",
        "SPX_ONLY",
        "FULL_NDX",
        "FULL_SPX",
    ]

    labels = {
        "NDX_ONLY": "Nasdaq seul",
        "SPX_ONLY": "S&P 500 seul",
        "FULL_NDX": "Modèle complet — Nasdaq",
        "FULL_SPX": "Modèle complet — S&P 500",
    }

    styles = {
        "NDX_ONLY": (HC_NAVY, 1.9, "-"),
        "SPX_ONLY": (HC_CHARCOAL, 1.7, "-"),
        "FULL_NDX": (HC_NAVY, 1.25, "--"),
        "FULL_SPX": (HC_MID, 1.25, "--"),
    }

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(
            figsize=(10.5, 5.5)
        )

        for factor_set in factor_sets:
            sample = current.loc[
                current["factor_set"]
                == factor_set
            ].sort_values("date")

            color, linewidth, linestyle = styles[
                factor_set
            ]

            ax.plot(
                pd.to_datetime(sample["date"]),
                sample["f_stat"].to_numpy(
                    dtype=float
                ),
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                label=labels[factor_set],
            )

        ax.axvline(
            cast(
                Any,
                pd.Timestamp("2020-03-09"),
            ),
            color=HC_MID,
            linestyle=":",
            linewidth=1.2,
        )

        ax.set_xlabel("Date candidate")
        ax.set_ylabel(r"Statistique $F(\tau)$")
        ax.set_title(
            "Diagnostic de changement selon la date candidate"
        )
        ax.grid(axis="y")
        ax.legend(
            ncol=2,
            loc="upper right",
        )

        ax.text(
            cast(
                Any,
                pd.Timestamp("2020-03-09"),
            ),
            ax.get_ylim()[1] * 0.97,
            "  maximum commun : 9 mars 2020",
            ha="left",
            va="top",
            fontsize=8.8,
            color=HC_MID,
        )

        fig.tight_layout()
        return _save(
            fig,
            PUBLICATION_FIGURES[7],
        )


def plot_09_evenements_meme_jour() -> Path:
    df = _require_table(
        "phase6b_corrected_event_responses.csv"
    )

    selected_events = [
        "NDX_DOWN_10",
        "RISK_OFF_2OF3",
        "RISK_ON_2OF3",
        "VIX_ORTHO_UP_10",
        "CREDIT_WIDEN_10",
    ]

    current = df.loc[
        (df["horizon"] == "SAME_DAY")
        & df["event"].isin(selected_events)
    ].copy()

    available = [
        event
        for event in selected_events
        if event in set(current["event"])
    ]

    pivot = (
        current.pivot(
            index="event",
            columns="period",
            values="mean_return",
        )
        .reindex(available)
        * 100.0
    )

    x = np.arange(len(available))
    width = 0.24
    palette = [
        HC_LIGHT,
        HC_CHARCOAL,
        HC_NAVY,
    ]

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(
            figsize=(11.0, 5.8)
        )

        for idx, period in enumerate(PERIODS):
            ax.bar(
                x + (idx - 1) * width,
                pivot[period].to_numpy(
                    dtype=float
                ),
                width,
                color=palette[idx],
                label=PERIOD_LABELS[period],
            )

        ax.axhline(
            0,
            linewidth=0.9,
            color=HC_CHARCOAL,
        )

        ax.set_xticks(
            x,
            [EVENT_LABELS[e] for e in available],
            rotation=18,
            ha="right",
        )
        ax.set_ylabel("Rendement moyen de Bitcoin (%)")
        ax.set_title(
            "Réponse contemporaine de Bitcoin aux événements extrêmes"
        )
        ax.grid(axis="y")
        ax.legend(
            ncol=3,
            loc="upper left",
        )

        ax.text(
            0.01,
            -0.24,
            "Résultats descriptifs : aucune différence de la famille "
            "événementielle ne survit à la correction FDR globale à 5 %.",
            transform=ax.transAxes,
            fontsize=8.8,
            color=HC_MID,
        )

        fig.tight_layout()
        return _save(
            fig,
            PUBLICATION_FIGURES[8],
        )


def plot_10_evenements_horizons() -> Path:
    df = _require_table(
        "phase6b_corrected_difference_tests.csv"
    )

    selected_events = [
        "NDX_DOWN_10",
        "NDX_UP_10",
        "RISK_OFF_2OF3",
        "RISK_ON_2OF3",
        "VIX_ORTHO_UP_10",
        "CREDIT_WIDEN_10",
    ]

    horizons = [
        "SAME_DAY",
        "FWD_1",
        "FWD_5",
        "FWD_20",
    ]

    current = df.loc[
        (df["comparison"] == "P3_MINUS_P1")
        & df["event"].isin(selected_events)
        & df["horizon"].isin(horizons)
    ].copy()

    available = [
        event
        for event in selected_events
        if event in set(current["event"])
    ]

    values = (
        current.pivot(
            index="event",
            columns="horizon",
            values="mean_difference",
        )
        .reindex(
            index=available,
            columns=horizons,
        )
        * 100.0
    )

    q_values = (
        current.pivot(
            index="event",
            columns="horizon",
            values="q_value_bh",
        )
        .reindex(
            index=available,
            columns=horizons,
        )
    )

    rejects = (
        current.pivot(
            index="event",
            columns="horizon",
            values="reject_fdr_5pct",
        )
        .reindex(
            index=available,
            columns=horizons,
        )
        .fillna(False)
        .astype(bool)
    )

    matrix = values.to_numpy(dtype=float)

    finite = np.abs(
        matrix[np.isfinite(matrix)]
    )

    vmax = (
        float(finite.max())
        if finite.size
        else 1.0
    )

    vmax = max(vmax, 0.5)

    cmap = LinearSegmentedColormap.from_list(
        "hilmar_diverging",
        [HC_CHARCOAL, HC_WHITE, HC_NAVY],
    )

    with plt.rc_context(_style()):
        fig, ax = plt.subplots(
            figsize=(9.6, 6.0)
        )

        image = ax.imshow(
            matrix,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            aspect="auto",
        )

        ax.set_xticks(
            np.arange(len(horizons)),
            [HORIZON_LABELS[h] for h in horizons],
        )
        ax.set_yticks(
            np.arange(len(available)),
            [EVENT_LABELS[e] for e in available],
        )

        for i, event in enumerate(available):
            for j, horizon in enumerate(horizons):
                value = cast(
                    Any,
                    values.loc[event, horizon],
                )
                q_value = cast(
                    Any,
                    q_values.loc[
                        event,
                        horizon,
                    ],
                )
                rejected = bool(
                    cast(
                        Any,
                        rejects.loc[event, horizon],
                    )
                )

                if pd.isna(value):
                    label = "—"
                else:
                    q_text = (
                        "NA"
                        if pd.isna(q_value)
                        else f"{float(q_value):.2f}"
                    )
                    star = "*" if rejected else ""
                    label = (
                        f"{float(value):+.1f}{star}\n"
                        f"q={q_text}"
                    )

                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8.4,
                    color=(
                        HC_WHITE
                        if (
                            pd.notna(value)
                            and abs(float(value))
                            > 0.58 * vmax
                        )
                        else HC_CHARCOAL
                    ),
                )

        ax.set_title(
            "Événements et horizons futurs — différence P3 moins P1"
        )
        ax.set_xlabel("Horizon")
        ax.set_ylabel("")

        colorbar = fig.colorbar(
            image,
            ax=ax,
            fraction=0.035,
            pad=0.03,
        )
        colorbar.set_label(
            "Différence de rendement moyen (points de %)"
        )

        ax.text(
            0.0,
            -0.13,
            "Une étoile serait affichée uniquement pour un rejet après correction "
            "Benjamini-Hochberg à 5 %.",
            transform=ax.transAxes,
            fontsize=8.8,
            color=HC_MID,
        )

        fig.tight_layout()
        return _save(
            fig,
            PUBLICATION_FIGURES[9],
        )


def generate_publication_figures() -> list[Path]:
    paths = [
        plot_01_synchronisation_temporelle(),
        plot_02_r2_ajuste_par_periode(),
        plot_03_r2_roulant_252(),
        plot_04_beta_nasdaq_par_periode(),
        plot_05_betas_roulants_252(),
        plot_06_shapley_r2_absolu(),
        plot_07_nasdaq_vs_sp500(),
        plot_08_profils_rupture(),
        plot_09_evenements_meme_jour(),
        plot_10_evenements_horizons(),
    ]

    print(
        "PUBLICATION_FIGURES=PASS"
    )
    print(
        "PUBLICATION_FIGURE_COUNT="
        + str(len(paths))
    )

    for path in paths:
        print(
            "PUBLICATION_FIGURE="
            + str(path)
        )

    return paths


def main() -> int:
    generate_publication_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

