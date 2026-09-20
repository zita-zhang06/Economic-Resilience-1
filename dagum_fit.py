from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import xlrd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from openpyxl.styles import Alignment

DESKTOP = Path(r"C:\Users\lironghua\Desktop")
SOURCE = DESKTOP / "publication data.xls"
OUTPUT = DESKTOP / "dagum_results"
COUNTRIES = ["Brazil", "India"]
SUBTYPES = ["Flash flood", "Flood (General)", "Riverine flood"]
YEARS = 25


def read_data(file):
    events = pd.read_excel(file, sheet_name="EMdat", engine="xlrd")
    gdp = pd.read_excel(file, sheet_name="GDP", engine="xlrd")
    events.columns = events.columns.astype(str).str.strip()
    gdp.columns = gdp.columns.astype(str).str.strip()
    sheet = xlrd.open_workbook(file, formatting_info=True).sheet_by_name("EMdat")
    visible = np.array([
        not sheet.rowinfo_map[i + 1].hidden if i + 1 in sheet.rowinfo_map else True
        for i in range(len(events))
    ])
    for col in ["Country", "Disaster Type", "Disaster Subtype"]:
        events[col] = events[col].astype(str).str.strip()
    damage = "Total Damage ('000 US$)"
    for col in [damage, "Start Year"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")
    selected = (
        events.Country.isin(COUNTRIES)
        & events["Disaster Type"].eq("Flood")
        & events["Disaster Subtype"].isin(SUBTYPES)
        & events["Start Year"].between(2000, 2024)
        & events[damage].gt(0)
    )
    if not np.array_equal(visible, selected.to_numpy()):
        raise ValueError("Excel visible rows do not match the sample. Check the filters.")
    data = events.loc[selected].copy()
    if data.groupby("Country").size().to_dict() != {"Brazil": 51, "India": 68}:
        raise ValueError("Expected 51 Brazil events and 68 India events. Check the file.")
    if data["DisNo."].duplicated().any():
        raise ValueError("Duplicate event IDs in the sample.")
    data["Excel row"] = data.index + 2
    data["GDP (USD)"] = np.nan
    for country in COUNTRIES:
        row = gdp.loc[gdp["Country Name"].astype(str).str.strip().eq(country)]
        if len(row) != 1 or str(row.iloc[0]["Indicator Name"]).strip() != "GDP (current US$)":
            raise ValueError(f"Check the GDP row and currency units for {country}.")
        annual = {int(y): float(row.iloc[0][y]) for y in gdp.columns if y.isdigit()}
        mask = data.Country.eq(country)
        data.loc[mask, "GDP (USD)"] = data.loc[mask, "Start Year"].map(annual)
    if not (np.isfinite(data["GDP (USD)"]) & data["GDP (USD)"].gt(0)).all():
        raise ValueError("Missing or invalid GDP values.")

    data["Damage (USD)"] = data[damage] * 1000
    data["Loss/GDP"] = data["Damage (USD)"] / data["GDP (USD)"]
    columns = ["Country", "DisNo.", "Start Year", "Disaster Subtype", "Excel row",
               damage, "GDP (USD)", "Damage (USD)", "Loss/GDP"]
    return data[columns].sort_values(["Country", "Start Year", "DisNo."])


def dagum_survival(x, a, b, p):
    log_cdf = -p * np.logaddexp(0, a * (np.log(b) - np.log(x)))
    return -np.expm1(log_cdf)


def fit_dagum(x):
    scale = np.median(x)
    log_x = np.log(x / scale)
    bounds = [(-9, 9), (-9, 9), (-25, 25)]

    def negative_loglik(q):
        a, p = np.exp(q[:2])
        z = a * (q[2] - log_x)
        log_density = q[0] + q[1] - log_x + z - (p + 1) * np.logaddexp(0, z)
        return -log_density.sum()

    rows = []
    for a0 in [0.5, 1.5, 3.0]:
        for p0 in [0.1, 1.0, 10.0]:
            log_b0 = np.log(np.expm1(np.log(2) / p0)) / a0
            start = [np.log(a0), np.log(p0), log_b0]
            fit = minimize(
                negative_loglik, start, method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7}
            )
            a, p = np.exp(fit.x[:2])
            b = np.exp(fit.x[2]) * scale
            boundary = any(min(q - lo, hi - q) < 0.001
                           for q, (lo, hi) in zip(fit.x, bounds))
            rows.append({
                "Start a": a0, "Start p": p0, "a": a, "b": b, "p": p,
                "Log likelihood": -fit.fun - len(x) * np.log(scale),
                "Converged": bool(fit.success), "At boundary": boundary,
                "Message": str(fit.message)
            })
    starts = pd.DataFrame(rows)
    good = starts.loc[starts.Converged & np.isfinite(starts["Log likelihood"])]
    if good.empty:
        raise RuntimeError("No successful Dagum fit. See the data and fitting settings.")
    best = good.loc[good["Log likelihood"].idxmax()]
    starts["Loglik gap"] = best["Log likelihood"] - starts["Log likelihood"]
    return best, starts


def analyse(data):
    summaries, all_starts = [], []
    for country in COUNTRIES:
        x = np.sort(data.loc[data.Country.eq(country), "Loss/GDP"].to_numpy())
        best, starts = fit_dagum(x)
        a, b, p = best[["a", "b", "p"]].to_numpy(dtype=float)
        n = len(x)
        cdf = 1 - dagum_survival(x, a, b, p)
        ks = max(np.max(np.arange(1, n + 1) / n - cdf),
                 np.max(cdf - np.arange(n) / n))
        loglik = best["Log likelihood"]
        near_best = starts.Converged & starts["Loglik gap"].abs().lt(1e-5)
        summaries.append({
            "Country": country, "Events": n, "Rate /year": n / YEARS,
            "a": a, "b": b, "p": p, "KS": ks,
            "Log likelihood": loglik, "AIC": 6 - 2 * loglik,
            "BIC": 3 * np.log(n) - 2 * loglik,
            "Successful starts": int(starts.Converged.sum()),
            "Starts near best": int(near_best.sum()),
            "At boundary": bool(best["At boundary"]),
            "Finite fitted mean": bool(a > 1), "Finite fitted variance": bool(a > 2)
        })
        starts.insert(0, "Country", country)
        all_starts.append(starts)
        print(f"{country}: n={n}, a={a:.6f}, b={b:.8g}, p={p:.6f}, KS={ks:.6f}")
    return pd.DataFrame(summaries), pd.concat(all_starts, ignore_index=True)


def draw_figure(data, summary, folder):
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    low = data["Loss/GDP"].min() * 0.8
    high = data["Loss/GDP"].max() * 1.3
    grid = np.geomspace(low, high, 400)
    for country, color, marker, line in zip(
        COUNTRIES, ["#31688e", "#d97524"], ["o", "^"], ["-", "--"]
    ):
        x = np.sort(data.loc[data.Country.eq(country), "Loss/GDP"].to_numpy())
        values, first = np.unique(x, return_index=True)
        empirical = (len(x) - first) / len(x)
        fit = summary.loc[summary.Country.eq(country)].iloc[0]
        ax.scatter(values, empirical, s=24, marker=marker, color=color,
                   alpha=0.75, label=f"{country}: observed (n={len(x)})", zorder=3)
        ax.plot(grid, dagum_survival(grid, fit.a, fit.b, fit.p),
                color=color, linestyle=line, linewidth=1.8, label=f"{country}: Dagum fit")
        ax.scatter(values[-1], empirical[-1], s=125, marker="*",
                   color=color, edgecolors="black", linewidths=0.5, zorder=4)
    ax.scatter([], [], s=100, marker="*", color="gray", label="Largest event in each country")
    ax.set(xscale="log", yscale="log", xlim=(low, high), ylim=(1e-4, 1.1),
           xlabel="Event loss / annual GDP", ylabel="Exceedance probability, P(X >= x)",
           title="Flood-loss distributions, 2000-2024")
    ax.grid(which="major", color="0.85", linewidth=0.6)
    ax.grid(which="minor", color="0.93", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    for extension in ["png", "pdf"]:
        fig.savefig(folder / f"Dagum_comparison.{extension}", dpi=300)
    plt.close(fig)


def save_excel(data, summary, starts, folder):
    notes = pd.DataFrame([
        ["Input", str(SOURCE)],
        ["Sample", "Visible positive-loss flood records, 2000-2024: Brazil 51, India 68"],
        ["Units", "Loss/GDP = Total Damage ('000 US$) * 1000 / GDP (current US$)"],
        ["CDF", "F(x) = [1 + (x/b)^(-a)]^(-p), x > 0; a, b, p > 0"],
        ["Survival", "S(x) = 1 - F(x)"],
        ["Parameters", "a: upper-tail shape; b: scale in GDP-share units; p: second shape"],
        ["Estimation", "MLE; location fixed at zero; median rescaling; 9 starting values"],
        ["Numerical bounds", "log(a), log(p): [-9,9]; log(b/median loss ratio): [-25,25]"],
        ["Convergence", "Near best: log-likelihood difference < 1e-5; not parameter uncertainty"],
        ["Rate", "Positive-loss records / 25 years; not the frequency of all floods"],
        ["KS", "Descriptive fitted-versus-empirical distance; no goodness-of-fit p-value"],
        ["Scope", "Dagum point estimates only; no Pareto threshold or bootstrap intervals"],
        ["Moments", "Unbounded fitted model: mean exists if a > 1; variance exists if a > 2"],
        ["Comparison", "Do not use AIC/BIC to rank fits across countries or different samples"],
        ["Figure", "Empirical exceedance and fitted survival; stars mark largest observed events"],
        ["Refresh", "Excel contains saved results; rerun Python after changing inputs"],
        ["Versions", f"numpy {np.__version__}; pandas {pd.__version__}; scipy {scipy.__version__}"]
    ], columns=["Item", "Details"])
    sheets = {"Fits": summary, "Starts": starts, "Events": data, "Notes": notes}
    with pd.ExcelWriter(folder / "Dagum_fit_results.xlsx", engine="openpyxl") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                header = column[0].value
                width = max(len(str(c.value or "")) for c in column) + 2
                sheet.column_dimensions[column[0].column_letter].width = min(width, 80)
                for cell in column[1:]:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.000000" if header != "b" else "0.000000E+00"
                        if header in ["Loss/GDP", "Loglik gap"]:
                            cell.number_format = "0.000000E+00"
                        elif header in ["GDP (USD)", "Damage (USD)", "Total Damage ('000 US$)"]:
                            cell.number_format = "#,##0.00"
                    elif isinstance(cell.value, str):
                        cell.alignment = Alignment(wrap_text=True, vertical="center")
            if name == "Notes":
                for row in range(2, sheet.max_row + 1):
                    sheet.row_dimensions[row].height = 30


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = read_data(SOURCE)
    summary, starts = analyse(data)
    draw_figure(data, summary, OUTPUT)
    save_excel(data, summary, starts, OUTPUT)
    print(f"Saved in: {OUTPUT}")


if __name__ == "__main__":
    main()
