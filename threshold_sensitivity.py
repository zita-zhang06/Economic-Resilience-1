from pathlib import Path
import numpy as np
import pandas as pd
import xlrd

DESKTOP = Path(r"C:\Users\lironghua\Desktop")
YEARS, PATHS, MONTHS, BOOT = 25, 5000, 600, 1000
THETA, DAMAGE_LIMIT, WINDOW = 1.0, 0.01, 60
SEED = 20260921
SUBTYPES = ["Flash flood", "Flood (General)", "Riverine flood"]

GRID = np.unique(np.round(np.r_[np.arange(1, 10)*1e-6,
                 np.arange(1, 21)*5e-6, np.arange(3, 21)*5e-5], 10))
PRIMARY = np.round(np.arange(1, 11)*1e-5, 10)


def load_sample(path):
    events = pd.read_excel(path, sheet_name="EMdat", engine="xlrd")
    gdp = pd.read_excel(path, sheet_name="GDP", engine="xlrd")
    events.columns = events.columns.astype(str).str.strip()
    gdp.columns = gdp.columns.astype(str).str.strip()

    book = xlrd.open_workbook(path, formatting_info=True)
    sheet = book.sheet_by_name("EMdat")
    visible = np.array([not sheet.rowinfo_map[k+1].hidden
               if k+1 in sheet.rowinfo_map else True for k in range(len(events))])

    for col in ["Country", "Disaster Type", "Disaster Subtype"]:
        events[col] = events[col].astype(str).str.strip()

    damage = "Total Damage ('000 US$)"
    for col in [damage, "Start Year"]:
        events[col] = pd.to_numeric(events[col], errors="coerce")

    mask = (events.Country.isin(["Brazil", "India"])
            & events["Disaster Type"].eq("Flood")
            & events["Disaster Subtype"].isin(SUBTYPES)
            & events["Start Year"].between(2000, 2024)
            & events[damage].gt(0))

    if not np.array_equal(visible, mask.to_numpy()):
        raise ValueError("Visible rows differ from the stated sample. Check Excel filters; do not silently include hidden rows.")

    selected = events.loc[visible].copy()
    counts = selected.groupby("Country").size().to_dict()
    if counts != {"Brazil": 51, "India": 68}:
        raise ValueError(f"Expected Brazil=51 and India=68; found {counts}. Check file version.")

    if selected["DisNo."].duplicated().any():
        raise ValueError("Duplicate disaster IDs in the selected sample.")

    selected["Source Excel row"] = selected.index + 2
    selected["GDP"] = np.nan
    gdp["Country Name"] = gdp["Country Name"].astype(str).str.strip()

    for country in ["Brazil", "India"]:
        row = gdp.loc[gdp["Country Name"].eq(country)]
        if len(row) != 1:
            raise ValueError(f"GDP country key is not unique: {country}")
        lookup = {int(y): float(row.iloc[0][y]) for y in gdp.columns if y.isdigit()}
        sel = selected.Country.eq(country)
        selected.loc[sel, "GDP"] = selected.loc[sel, "Start Year"].map(lookup)

    if not (np.isfinite(selected.GDP) & selected.GDP.gt(0)).all():
        raise ValueError("Missing or invalid GDP denominator.")

    selected["ratio"] = selected[damage] * 1000 / selected.GDP
    return selected.sort_values(["Country", "Start Year", "DisNo."]), int(visible.sum())


def pareto_fit(x, u, seed):
    tail = np.sort(x[x >= u])
    n = len(tail)
    if n < 2:
        return n, None, None, None, None

    logs = np.log(tail/u)
    alpha = n/logs.sum()
    cdf = -np.expm1(-alpha*logs)
    ks = max(np.max(np.arange(1, n+1)/n-cdf), np.max(cdf-np.arange(n)/n))

    rng = np.random.default_rng(seed)
    means = rng.choice(logs, size=(BOOT, n), replace=True).mean(axis=1)
    estimates = 1/means[means > 0]
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return n, float(alpha), float(lo), float(hi), float(ks)


def make_common_events(n_positive, seed):
    rng = np.random.default_rng(seed)
    months = []
    for _ in range(MONTHS):
        counts = rng.poisson(n_positive/YEARS/12, PATHS)
        index = np.repeat(np.arange(PATHS), counts)
        months.append((index, rng.random(len(index)), rng.exponential(size=len(index))))
    return months


def simulate(n, total_n, u, alpha, months):
    current = np.zeros(PATHS)
    maximum = np.zeros(PATHS)
    last_sum = np.zeros(PATHS)

    for t, (index, keep_uniform, exp_draw) in enumerate(months):
        keep = keep_uniform < n/total_n
        loss = np.exp(np.minimum(np.log(u)+exp_draw[keep]/alpha, 0.0))
        jumps = np.bincount(index[keep], weights=loss, minlength=PATHS)
        current = np.clip(current*(1-THETA/12)+jumps, 0, 1)
        maximum = np.maximum(maximum, current)
        if t >= MONTHS-WINDOW:
            last_sum += current

    p = float(np.mean(last_sum/WINDOW >= DAMAGE_LIMIT))
    z = 1.959963984540054
    den = 1+z*z/PATHS
    center = (p+z*z/(2*PATHS))/den
    half = z*np.sqrt(p*(1-p)/PATHS+z*z/(4*PATHS**2))/den

    return [float(maximum.mean()), float(np.median(maximum)),
            float(np.quantile(maximum, .95)), float(np.mean(maximum >= DAMAGE_LIMIT)),
            p, float(center-half), float(center+half), float(np.mean(maximum >= 1))]


def analyse(data):
    rows = []
    for country_index, country in enumerate(["Brazil", "India"]):
        x = data.loc[data.Country.eq(country), "ratio"].to_numpy()
        months = make_common_events(len(x), SEED + country_index)

        for j, u in enumerate(GRID):
            fit_seed = SEED + 1000 * (country_index + 1) + j
            n, alpha, low, high, ks = pareto_fit(x, u, fit_seed)
            risk = simulate(n, len(x), u, alpha, months) if n >= 5 else [None] * 8
            grid = "Main" if u in PRIMARY else "Fine" if 1e-5 <= u <= 1e-4 else "Stress"
            note = "Tail sample below 20" if n < 20 else ""
            if country == "Brazil" and u == 2e-5:
                note = "Eric's original baseline"

            rows.append([
                country, grid, u, n, n / len(x), n / YEARS,
                alpha, low, high, ks, *risk, note
            ])
            print(f"{country}: {j + 1}/{len(GRID)}, threshold={u:g}, events={n}", flush=True)

    columns = [
        "Country", "Grid", "Threshold", "Tail events", "Retained share",
        "Arrival rate /year", "Alpha", "Alpha lower", "Alpha upper", "KS",
        "Mean maximum damage", "Median maximum damage", "Maximum damage P95",
        "Any exceedance probability", "Persistent damage probability",
        "MC lower", "MC upper", "Cap reached probability", "Note"
    ]
    results = pd.DataFrame(rows, columns=columns)

    for country in ["Brazil", "India"]:
        eligible = results.loc[
            results.Country.eq(country) & results.Grid.eq("Main")
            & results["Tail events"].ge(20)
        ]
        if not eligible.empty:
            best = eligible.KS.idxmin()
            results.loc[best, "Note"] += "; Lowest main-grid KS; candidate only"
    results["Note"] = results["Note"].str.strip("; ")

    shortlist = np.round(np.array([2, 3, 3.5, 4, 4.5, 5, 6]) * 1e-5, 10)
    review = results.loc[results.Threshold.isin(shortlist), [
        "Country", "Threshold", "Tail events", "Retained share",
        "Arrival rate /year", "Alpha", "KS", "Persistent damage probability",
        "Any exceedance probability", "Note"
    ]].copy()
    review["Eric comments"] = ""
    return results, review


def save_excel(data, results, review, source, output):
    events = data[[
        "Country", "DisNo.", "Source Excel row", "Disaster Subtype",
        "Start Year", "Total Damage ('000 US$)", "GDP", "ratio"
    ]].rename(columns={"ratio": "Loss / GDP"})

    settings = pd.DataFrame([
        ["Source", str(source)],
        ["Sample", "Visible rows: Brazil 51, India 68; coastal flood already excluded"],
        ["Observation period", "2000-2024; arrival rate = tail events / 25 years"],
        ["Loss ratio", "Damage in thousand current USD * 1000 / annual GDP in current USD"],
        ["Threshold grid", "46 thresholds per country; Main = 1,2,...,10 times 1e-5"],
        ["Original baseline", "Brazil 2e-5, selected by Eric; KS candidates do not replace it"],
        ["Small samples", "n<20 flagged; n<5 simulation omitted; exploratory cutoffs"],
        ["Pareto fit", "MLE alpha; empirical-versus-fitted KS distance"],
        ["Alpha interval", f"95% percentile bootstrap; {BOOT} resamples; threshold held fixed"],
        ["Simulation", f"{PATHS} paths; {MONTHS} months; theta={THETA} per year"],
        ["Recovery", "D_next = clip(D * (1-theta/12) + monthly losses, 0, 1)"],
        ["Any exceedance", f"Maximum simulated damage >= {DAMAGE_LIMIT}"],
        ["Persistent damage", f"Mean damage in final {WINDOW} months >= {DAMAGE_LIMIT}"],
        ["Cap", "Individual losses capped at 1; aggregate damage clipped to [0,1]"],
        ["MC interval", "95% Wilson interval for persistent probability; simulation error only"],
        ["Random seed", SEED],
        ["Shared draws", "Same underlying random events and severity quantiles across thresholds"],
        ["Interpretation", "Conditional model experiment; loss/GDP enters recovery directly; not an economic forecast"],
        ["Model scope", "Pareto threshold sensitivity only; no Dagum refit or recovery-rate selection"],
        ["Refresh", "Rerun Python after changing data or settings; Excel contains saved results"],
        ["Packages", f"numpy {np.__version__}; pandas {pd.__version__}; xlrd {xlrd.__version__}"]
    ], columns=["Setting", "Value"])

    sheets = {"Review": review, "Results": results, "Events": events, "Settings": settings}
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, table in sheets.items():
            table.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions

            for column in sheet.columns:
                header = column[0].value
                width = max(len(str(cell.value or "")) for cell in column) + 2
                sheet.column_dimensions[column[0].column_letter].width = min(width, 75)
                for cell in column[1:]:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0000"
                        if header in ["Threshold", "Loss / GDP"]:
                            cell.number_format = "0.00E+00"
                        elif header == "Retained share" or "probability" in header or header in ["MC lower", "MC upper"]:
                            cell.number_format = "0.00%"


def main():
    source = DESKTOP / "publication data.xls"
    output = DESKTOP / "Flood_threshold_sensitivity.xlsx"
    print("Reading Excel...", flush=True)
    data, count = load_sample(source)
    print(f"Selected events: {count}", flush=True)
    results, review = analyse(data)
    save_excel(data, results, review, source, output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
