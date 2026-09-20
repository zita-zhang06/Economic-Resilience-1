# -*- coding: utf-8 -*-
"""Run in CMD: python threshold_sensitivity.py
Dependencies: python -m pip install numpy pandas scipy xlrd openpyxl
Only reads the source XLS; writes a separate XLSX and an audit JSON.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import xlrd

DESKTOP = Path(r"C:\Users\lironghua\Desktop")
YEARS, PATHS, MONTHS, BOOT = 25, 5000, 600, 1000
THETA, DAMAGE_LIMIT, WINDOW = 1.0, 0.01, 60
SEED = 20260921
SUBTYPES = ["Flash flood", "Flood (General)", "Riverine flood"]
# 46 distinct regular thresholds. The primary grid is 1,2,...,10 x 10^-5.
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
    tail = np.sort(x[x >= u]); n = len(tail)
    if n < 2:
        return n, None, None, None, None
    logs = np.log(tail/u); alpha = n/logs.sum()
    cdf = -np.expm1(-alpha*logs)
    ks = max(np.max(np.arange(1, n+1)/n-cdf), np.max(cdf-np.arange(n)/n))
    rng = np.random.default_rng(seed)
    means = rng.choice(logs, size=(BOOT, n), replace=True).mean(axis=1)
    estimates = 1/means[means > 0]
    lo, hi = np.percentile(estimates, [2.5, 97.5])
    return n, float(alpha), float(lo), float(hi), float(ks)


def make_common_events(n_positive, seed):
    rng = np.random.default_rng(seed); months = []
    for _ in range(MONTHS):
        counts = rng.poisson(n_positive/YEARS/12, PATHS)
        index = np.repeat(np.arange(PATHS), counts)
        months.append((index, rng.random(len(index)), rng.exponential(size=len(index))))
    return months


def simulate(n, total_n, u, alpha, months):
    current = np.zeros(PATHS); maximum = np.zeros(PATHS); last_sum = np.zeros(PATHS)
    for t, (index, keep_uniform, exp_draw) in enumerate(months):
        keep = keep_uniform < n/total_n
        loss = np.exp(np.minimum(np.log(u)+exp_draw[keep]/alpha, 0.0))
        jumps = np.bincount(index[keep], weights=loss, minlength=PATHS)
        current = np.clip(current*(1-THETA/12)+jumps, 0, 1)
        maximum = np.maximum(maximum, current)
        if t >= MONTHS-WINDOW:
            last_sum += current
    p = float(np.mean(last_sum/WINDOW >= DAMAGE_LIMIT))
    z = 1.959963984540054; den = 1+z*z/PATHS
    center = (p+z*z/(2*PATHS))/den
    half = z*np.sqrt(p*(1-p)/PATHS+z*z/(4*PATHS**2))/den
    return [float(maximum.mean()), float(np.median(maximum)),
            float(np.quantile(maximum, .95)), float(np.mean(maximum >= DAMAGE_LIMIT)),
            p, float(center-half), float(center+half), float(np.mean(maximum >= 1))]


def analyse(path):
    data, visible_n = load_sample(path); rows = []; recommended = {}
    for cidx, country in enumerate(["Brazil", "India"]):
        x = data.loc[data.Country.eq(country), "ratio"].to_numpy()
        months = make_common_events(len(x), SEED+cidx)
        for j, u in enumerate(GRID):
            n, alpha, lo, hi, ks = pareto_fit(x, u, SEED+1000*(cidx+1)+j)
            category = ("Primary" if np.any(np.isclose(u, PRIMARY, rtol=0, atol=1e-12))
                        else "Fine" if 1e-5 <= u <= 1e-4 else "Stress")
            risk = simulate(n, len(x), u, alpha, months) if n >= 5 else [None]*8
            rows.append([country, category, float(u), float(u), n, n/len(x), n/YEARS,
                         alpha, lo, hi, ks, None, *risk,
                         PATHS if n >= 5 else None, "", "Small tail: n<20" if n<20 else ""])
            print(f"{country}: threshold {j+1}/{len(GRID)} = {u:.7f}, tail n={n}", flush=True)
        eligible = sorted([r for r in rows if r[0]==country and r[1]=="Primary" and r[4]>=20], key=lambda r:r[10])
        recommended[country] = eligible[0][2]
        for rank, r in enumerate(eligible, 1):
            r[11] = rank
            if rank == 1: r[21] = "Recommended primary-grid baseline"
        for r in rows:
            if r[0]==country and r[2]==2e-5:
                r[21] = (r[21]+"; " if r[21] else "")+"Original Brazil threshold / common comparison"
        fine_best = min([r for r in rows if r[0]==country and r[4]>=20], key=lambda r:r[10])
        if fine_best[2] != recommended[country]:
            fine_best[21] = "Lowest KS on the full regular grid; alternative for Eric"
    headers = ["Country", "Grid", "Threshold: GDP share", "Threshold: % GDP", "Tail events n",
        "Share of positive events", "Arrival rate /year", "Pareto alpha", "Alpha bootstrap low",
        "Alpha bootstrap high", "KS distance", "Primary-grid KS rank", "Mean maximum damage",
        "Median maximum damage", "95th pct maximum", "Pr(max >=1%)", "Pr(persistent >=1%)",
        "MC Wilson low", "MC Wilson high", "Pr(100% cap)", "MC paths", "Decision note", "Sample warning"]
    shortlist = np.array([2, 3, 3.5, 4, 4.5, 5, 6])*1e-5
    review_h = ["Country", "Threshold: GDP share", "Threshold: % GDP", "Tail events n",
               "Arrival rate /year", "Pareto alpha", "KS distance", "Pr(persistent >=1%)",
               "Pr(max >=1%)", "Decision note", "Eric selection / comments"]
    review = [[r[0], r[2], r[3], r[4], r[6], r[7], r[10], r[16], r[15], r[21], ""]
              for r in rows if np.any(np.isclose(r[2], shortlist, rtol=0, atol=1e-12))]
    event_h = ["Country", "Disaster ID", "Source Excel row", "Subtype", "Start year",
               "Damage: thousand current USD", "GDP: current USD", "Loss / GDP"]
    events = [[r["Country"], r["DisNo."], int(r["Source Excel row"]), r["Disaster Subtype"],
               int(r["Start Year"]), float(r["Total Damage ('000 US$)"]), float(r["GDP"]), float(r["ratio"])]
              for _, r in data.iterrows()]
    notes = [
      ["Observation years", YEARS, "2000-2024 inclusive", "Arrival rate denominator"],
      ["Visible selected records", visible_n, "Brazil 51 + India 68 = 119", "Hidden rows are excluded and audited"],
      ["Flood subtypes", "; ".join(SUBTYPES), "Coastal flood already excluded", "No change to the user's selection"],
      ["Primary threshold grid", "1,2,...,10 x 10^-5", "Ten regularly spaced main candidates", "Rank by KS among n>=20; grid definition is an analysis choice"],
      ["Fine threshold grid", "0.5 x 10^-5 spacing up to 10^-4", "Checks between main candidates", "All 46 distinct candidates retained; no rounding of fitted alpha"],
      ["Stress grid", "1-9 x 10^-6; 1.5-10 x 10^-4", "Extends both sides of the main range", "Sparse tails are flagged, not presented as reliable baselines"],
      ["Minimum recommended tail", 20, "Exploratory safeguard, not a universal statistical standard", "n<5 simulations not computed"],
      ["Rebuilding rate theta", THETA, "per year", "Same baseline as the dissertation; fixed in this threshold experiment"],
      ["Simulation horizon", MONTHS/12, "years; monthly time step", "D_next = clip(D*(1-theta/12)+J,0,1)"],
      ["Simulation paths", PATHS, "per country / threshold", "Poisson thinning and shared severity quantiles across thresholds"],
      ["Persistent damage threshold", DAMAGE_LIMIT, "model damage ratio", "Mean damage in final 60 months >=0.01"],
      ["Bootstrap resamples", BOOT, "conditional on each fixed threshold", "Nonparametric percentile interval for alpha; selection uncertainty not included"],
      ["Base random seed", SEED, "deterministic for a fixed dependency version", "Country and row-specific offsets are in the supplied Python code"],
      ["MC interval", "95% Wilson", "Simulation sampling uncertainty only", "Does not include parameter, reporting or model uncertainty"],
      ["Interpretation", "Conditional sensitivity experiment", "No threshold chosen to reach a desired collapse probability", "KS fit and tail counts inform selection; simulation measures consequences"],
      ["Units boundary", "Loss/GDP enters the original damage recursion directly", "Preserves dissertation convention", "GDP-to-capital conversion not newly calibrated; not an economic forecast"],
      ["Original threshold evidence", "Fixed 2e-5 + visual rationale + sensitivity checks", "Original checks: 1,1.5,2,2.5,3,4,5 x 10^-5", "No automatic original-selection routine found in archived code"],
      ["Input file", str(path), "SHA256 below", "A snapshot of the attached workbook; rerun if data change"],
      ["Input SHA256", hashlib.sha256(path.read_bytes()).hexdigest(), "File identity", ""],
      ["Versions", f"Python {platform.python_version()}; numpy {np.__version__}; pandas {pd.__version__}; xlrd {xlrd.__version__}", "Reproducibility", ""],
      ["Method reference", "Clauset, Shalizi & Newman (2009)", "https://doi.org/10.1137/070710111", "KS threshold diagnostics; this regular-grid variant is specified above"],
      ["Archived code", "https://github.com/zita-zhang06/dagum.file", "Appendix Table D2.txt", "Original model settings; new common-random-number simulation"],
      ["Refresh", "Rerun threshold_sensitivity.py", "Fitted and simulated results are saved snapshots", "Excel formulas show unit conversions and arrival rates; edits alone do not rerun simulations"]]
    sheets = {
      "Review": {"title":"Brazil and India | threshold review", "note":"46 thresholds per country; rounded primary-grid recommendations. See Methods for assumptions.", "headers":review_h, "rows":review},
      "Sensitivity": {"title":"Threshold sensitivity | 92 country-threshold combinations", "note":"Primary + fine + stress grids. Percent columns use Excel percentage format; Monte Carlo intervals exclude parameter uncertainty.", "headers":headers, "rows":rows},
      "Events": {"title":"Selected positive-loss events | 119 records", "note":"Source: publication data.xls, visible EMdat rows; contemporaneous GDP from GDP sheet. Original measurements preserved.", "headers":event_h, "rows":events},
      "Methods": {"title":"Sample audit and reproducible settings", "note":"Analysis snapshots; rerun the Python script after changing the source data or settings.", "headers":["Setting / evidence", "Value", "Units / context", "Explanation"], "rows":notes}}
    return {"sheets":sheets, "recommendations":recommended}


def write_excel(payload, output):
    # Standalone Windows export: openpyxl is available through pip.
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    wb = Workbook(); wb.remove(wb.active)
    for name, spec in payload["sheets"].items():
        ws = wb.create_sheet(name); ws.sheet_view.showGridLines = False
        ws["A2"] = spec["title"]; ws["A2"].font = Font(name="Calibri", size=15, bold=True)
        ws["A3"] = spec["note"]; ws["A3"].font = Font(name="Calibri", size=10, italic=True)
        for j, val in enumerate(spec["headers"], 1): ws.cell(5,j,val)
        for i, row in enumerate(spec["rows"], 6):
            for j, val in enumerate(row, 1):
                c = ws.cell(i,j,val); c.font = Font(name="Calibri",size=11)
                c.alignment = Alignment(vertical="center")
                if isinstance(val, float): c.number_format = "0.0000"
            if name=="Sensitivity":
                ws.cell(i,4,f"=C{i}"); ws.cell(i,6,f'=E{i}/COUNTIFS(Events!$A$6:$A$124,A{i})')
                ws.cell(i,7,f"=E{i}/Methods!$B$6")
            elif name=="Events": ws.cell(i,8,f"=F{i}*1000/G{i}")
            elif name=="Review":
                source = next(k+6 for k,r in enumerate(payload["sheets"]["Sensitivity"]["rows"]) if r[0]==row[0] and r[2]==row[1])
                for col,src in enumerate(["A","C","D","E","G","H","K","Q","P","V"],1):
                    ref = f"Sensitivity!{src}{source}"
                    ws.cell(i,col,f'=IF({ref}="","",{ref})' if src=="V" else f"={ref}")
        percentage_cols = {"Review":[3,8,9], "Sensitivity":[4,6,13,14,15,16,17,18,19,20]}.get(name,[])
        sci_cols = {"Review":[2], "Sensitivity":[3], "Events":[8]}.get(name,[])
        for col in percentage_cols+sci_cols:
            for i in range(6,ws.max_row+1): ws.cell(i,col).number_format = "0.0000%" if col in percentage_cols else "0.00E+00"
        for cell in ws[5]:
            cell.fill=PatternFill("solid",fgColor="263D58"); cell.font=Font(name="Calibri",bold=True,color="FFFFFF",size=11)
            cell.alignment=Alignment(wrap_text=True,horizontal="center",vertical="center")
        ws.row_dimensions[5].height=44; ws.freeze_panes="C6"; ws.auto_filter.ref=f"A5:{get_column_letter(ws.max_column)}{ws.max_row}"
        for j, head in enumerate(spec["headers"],1):
            ws.column_dimensions[get_column_letter(j)].width = 22 if len(head)>16 else 18
        if name=="Review": ws.column_dimensions["J"].width=57; ws.column_dimensions["K"].width=32
        if name=="Sensitivity": ws.column_dimensions["V"].width=57; ws.column_dimensions["W"].width=25
        if name=="Methods":
            for col,width in zip("ABCD",[32,65,60,85]): ws.column_dimensions[col].width=width
            for row in ws.iter_rows(min_row=6):
                for cell in row: cell.alignment=Alignment(wrap_text=True,vertical="center")
            for i in range(6,ws.max_row+1): ws.row_dimensions[i].height=44
        for i,row in enumerate(spec["rows"],6):
            if any(isinstance(v,str) and "Recommended primary-grid" in v for v in row):
                for cell in ws[i]: cell.fill=PatternFill("solid",fgColor="E4EFDA")
    wb.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DESKTOP/"publication data.xls")
    parser.add_argument("--output", type=Path, default=DESKTOP/"Flood_threshold_sensitivity.xlsx")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    result = analyse(args.input)
    audit = args.output.with_suffix(".json")
    audit.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if not args.json_only: write_excel(result,args.output)
    print("Recommended primary-grid thresholds:", result["recommendations"])
    print("Saved:", audit if args.json_only else args.output)
