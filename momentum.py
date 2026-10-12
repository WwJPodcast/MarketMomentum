#!/usr/bin/env python3
"""S&P 500 momentum card (price gain only, dividends excluded).

Usage:
  python momentum.py --out index.html --json data.json   # live data (needs internet)
  python momentum.py --demo --out demo.html              # synthetic data, offline test

Rules:
  * Rank S&P 500 stocks by 12-month and 6-month price gain as of the last close.
  * Checks on the 12-month top 3: in 6-month top 3, above 200-day average,
    within 15% of the 52-week high (closing prices).
  * BUY = passes all three, max 2 picks, at most one per sector.
  * NEW = in this 12-month top 3 but not in the previous report's (read from --json).
"""
import argparse
import datetime as dt
import io
import json
import os
from string import Template

import numpy as np
import pandas as pd

TOP_N = 3
NEAR_HIGH = 0.85   # within 15% of 52-week high
BUY_MAX = 2


# ---------- data loading ----------
def load_universe():
    import requests
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "sp-momentum/1.0"}, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    t["Symbol"] = t["Symbol"].str.replace(".", "-", regex=False)
    return t[["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})


def load_prices(tickers):
    import yfinance as yf
    # auto_adjust=False keeps Close split-adjusted but NOT dividend-adjusted = price gain only
    px = yf.download(tickers, period="15mo", auto_adjust=False,
                     progress=False, threads=True)["Close"]
    return px.dropna(how="all")


def demo_data(seed=7):
    rng = np.random.default_rng(seed)
    n = 30
    tickers = [f"DEMO{i + 1:02d}" for i in range(n)]
    sectors = ["Technology", "Health Care", "Energy", "Financials", "Industrials", "Utilities"]
    uni = pd.DataFrame({"ticker": tickers,
                        "name": [f"Demo Company {i + 1}" for i in range(n)],
                        "sector": [sectors[i % 6] for i in range(n)]})
    idx = pd.bdate_range(end=pd.Timestamp("2026-10-09"), periods=330)
    drift = rng.normal(0.0008, 0.0012, n)
    vol = rng.uniform(0.012, 0.03, n)
    rets = rng.normal(drift, vol, (len(idx), n))
    px = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=tickers)
    return uni, px


# ---------- calculations ----------
def compute(uni, px, asof=None):
    px = px.sort_index()
    if px.index.tz is not None:
        px.index = px.index.tz_localize(None)
    # never use today's (possibly intraday) bar: the report is always as of a completed close
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    px = px[px.index.normalize() < today]
    if asof:
        px = px[px.index <= pd.Timestamp(asof)]
    last = px.index[-1]
    d12 = last - pd.DateOffset(months=12)
    d6 = last - pd.DateOffset(months=6)
    rows = []
    for t in px.columns:
        s = px[t].dropna()
        if len(s) < 252 or s.index[0] > d12 or s.index[-1] != last:
            continue  # need a full year of history and a price on the as-of date
        c = s.iloc[-1]
        p12 = s[s.index <= d12].iloc[-1]
        p6 = s[s.index <= d6].iloc[-1]
        rows.append(dict(
            ticker=t, price=float(c),
            ret12=float(c / p12 - 1), ret6=float(c / p6 - 1),
            above200=bool(c > s.iloc[-200:].mean()),
            near_high=bool(c >= NEAR_HIGH * s.iloc[-252:].max())))
    df = pd.DataFrame(rows).merge(uni, on="ticker")
    top12 = df.nlargest(TOP_N, "ret12").reset_index(drop=True)
    top6 = df.nlargest(TOP_N, "ret6").reset_index(drop=True)
    six = set(top6["ticker"])
    top12["in6"] = top12["ticker"].isin(six)
    top12["passes"] = top12[["in6", "above200", "near_high"]].all(axis=1)
    top6["in12"] = top6["ticker"].isin(set(top12["ticker"]))
    buys, seen = [], set()
    for i, r in top12.iterrows():
        if r["passes"] and r["sector"] not in seen and len(buys) < BUY_MAX:
            buys.append(i)
            seen.add(r["sector"])
    return dict(asof=last, top12=top12, top6=top6, buys=buys)


def load_previous(path):
    """Tickers in the previous report's 12-month top 3 (empty set if none)."""
    try:
        with open(path, encoding="utf-8") as f:
            return {r["ticker"] for r in json.load(f)["top12"]}
    except (OSError, ValueError, KeyError):
        return set()


# ---------- rendering ----------
CHECK = ('<svg width="{s}" height="{s}" viewBox="0 0 16 16" role="img" aria-label="Pass">'
         '<path d="M3 8.5l3.2 3.2L13 4.8" fill="none" style="stroke:var(--green)" '
         'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
CROSS = ('<svg width="{s}" height="{s}" viewBox="0 0 16 16" role="img" aria-label="Fail">'
         '<path d="M4 4l8 8M12 4l-8 8" fill="none" style="stroke:var(--muted)" '
         'stroke-width="2" stroke-linecap="round"/></svg>')
NEW_CHIP = '<span class="new">NEW</span>'


def icon(ok, size=16):
    return (CHECK if ok else CROSS).format(s=size)


def pct(x):
    return f"{x * 100:+,.1f}%"


def usd(x):
    return f"${x:,.2f}"


def card(rank, r, is_new):
    checks = "".join(
        f'<div class="chk">{icon(ok, 14)}<span>{label}</span></div>'
        for label, ok in [("In 6-mo top 3", r.in6), ("Above 200-day", r.above200),
                          ("Within 15% of high", r.near_high)])
    chip = NEW_CHIP if is_new else ""
    return f"""<div class="card">
<div class="row"><span class="tags"><span class="tag">BUY · #{rank}</span>{chip}</span><span class="mono muted sm">{usd(r.price)}</span></div>
<div class="big">{r.ticker}</div><div class="muted sm">{r['name']}</div>
<div class="row mt"><span class="lab">12-MO</span><span class="mono g lg">{pct(r.ret12)}</span></div>
<div class="row"><span class="lab">6-MO</span><span class="mono g md">{pct(r.ret6)}</span></div>
<div class="chks">{checks}</div></div>"""


def render(res, demo=False, prev=frozenset()):
    t12, t6, buys, last = res["top12"], res["top6"], res["buys"], res["asof"]
    new = (set(t12["ticker"]) - set(prev)) if prev else set()
    if buys:
        body = '<div class="cards">' + "".join(
            card(n + 1, t12.loc[i], t12.loc[i, "ticker"] in new) for n, i in enumerate(buys)) + "</div>"
        head = "BUY · PASS ALL 3 CHECKS"
    else:
        body = '<div class="none">No buy this period. No name passed all three checks.</div>'
        head = "NO BUY THIS PERIOD"
    rows = ""
    for i, r in t12.iterrows():
        chip = NEW_CHIP if r.ticker in new else ""
        rows += (f'<div class="trow"><div class="tk"><span class="mono muted sm">{i + 1}</span>'
                 f'<b>{r.ticker}</b>{chip}</div><span class="mono g">{pct(r.ret12)}</span>'
                 f'<div class="c">{icon(r.in6)}</div><div class="c">{icon(r.above200)}</div>'
                 f'<div class="c">{icon(r.near_high)}</div></div>')
    cells = ""
    for i, r in t6.iterrows():
        bar = r.ret6 / t6["ret6"].max() * 100
        cls = "bar hi" if r.in12 else "bar"
        cells += (f'<div class="cell"><div class="tk"><span class="mono muted sm">{i + 1}</span><b>{r.ticker}</b></div>'
                  f'<div class="mono g">{pct(r.ret6)}</div>'
                  f'<div class="track"><div class="{cls}" style="width:{bar:.0f}%"></div></div></div>')
    pill = "Demo data (synthetic)" if demo else "Live data"
    return Template(PAGE).substitute(
        pill=pill, report=dt.date.today().strftime("%a, %b %-d, %Y"),
        asof=last.strftime("%a, %b %-d"), head=head, body=body, rows=rows, cells=cells)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Momentum leaders</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&family=Fraunces:wght@600&display=swap">
<style>
:root{--bg:#F4F1EA;--panel:#fff;--track:#E2DED2;--text:#1B1F2A;--muted:#5A5F6B;--line:#D9D4C7;--blue:#1F3FA6;--blue-bar:#1F3FA6;--strip:#1F3FA6;--green:#0B6B3A;--bar-muted:#8D93A0}
@media(prefers-color-scheme:dark){:root{--bg:#14171C;--panel:#1D2128;--track:#232830;--text:#ECEAE4;--muted:#A3A8B2;--line:#3A404B;--blue:#8FA8FF;--blue-bar:#6B8CFF;--strip:#3556D4;--green:#4CC585;--bar-muted:#5A6272}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif}
.wrap{max-width:390px;margin:0 auto;padding:26px 18px 20px;display:flex;flex-direction:column;gap:16px}
.mono{font-family:'DM Mono',monospace}.muted{color:var(--muted)}.g{color:var(--green)}.sm{font-size:12px}.md{font-size:15px}.lg{font-size:19px}
.eyebrow{font:11px 'DM Mono',monospace;letter-spacing:.14em;color:var(--muted)}
.top{display:flex;justify-content:space-between;align-items:center}
.pill{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:2px 8px}
h1{margin:6px 0;font-size:30px;line-height:1.1}.sub{font-size:13px;color:var(--muted)}
.sec{display:flex;flex-direction:column;gap:10px}.lbl{font:11px 'DM Mono',monospace;letter-spacing:.14em;color:var(--blue)}
.cards{display:flex;gap:10px}.card{flex:1;min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;display:flex;flex-direction:column;gap:6px}
.row{display:flex;justify-content:space-between;align-items:baseline}.mt{margin-top:4px}.tags{display:flex;gap:5px;align-items:center}
.tag{font:500 10px 'DM Mono',monospace;letter-spacing:.1em;background:var(--strip);color:#fff;border-radius:4px;padding:3px 7px}
.new{font:10px 'DM Mono',monospace;letter-spacing:.08em;background:var(--text);color:var(--bg);border-radius:4px;padding:2px 5px}
.big{font:600 40px/1 'Fraunces',serif;margin-top:4px}.lab{font:10px 'DM Mono',monospace;letter-spacing:.1em;color:var(--muted)}
.chks{display:flex;flex-direction:column;gap:5px;margin-top:6px;padding-top:8px;border-top:1px solid var(--line)}
.chk{display:flex;align-items:center;gap:7px;font:10px 'DM Mono',monospace}.none{border:1px dashed var(--line);border-radius:10px;padding:16px;color:var(--muted);font-size:14px}
.thead,.trow{display:grid;grid-template-columns:96px 80px 1fr 1fr 1fr;gap:4px;align-items:center}
.thead div{font:9px/1.3 'DM Mono',monospace;color:var(--muted);text-align:center}.thead div:nth-child(2){text-align:left}
.trow{padding:8px 0;border-top:1px solid var(--line)}.trow:last-child{border-bottom:1px solid var(--line)}
.tk{display:flex;align-items:baseline;gap:7px}.tk b{font-size:17px}.c{display:flex;justify-content:center}
.cells{display:flex;gap:14px}.cell{flex:1;min-width:0;display:flex;flex-direction:column;gap:5px}.cell .mono{font-size:13px}
.track{height:6px;background:var(--track);border-radius:3px}.bar{height:6px;border-radius:3px;background:var(--bar-muted)}.bar.hi{background:var(--blue-bar)}
.legend{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--muted)}.legend i{width:10px;height:10px;border-radius:2px;background:var(--blue-bar)}
.foot{border-top:1px solid var(--line);padding-top:10px;font-size:11px;line-height:1.45;color:var(--muted);display:flex;flex-direction:column;gap:6px}
</style></head><body><div class="wrap">
<div><div class="top"><div class="eyebrow">S&amp;P 500 MOMENTUM</div><div class="pill">$pill</div></div>
<h1>Momentum leaders</h1><div class="sub">$report · prices as of $asof close</div></div>
<div class="sec"><div class="row"><div class="lbl">$head</div><div class="mono muted sm">$$100–$$200 total</div></div>$body</div>
<div><div class="eyebrow" style="margin-bottom:6px">12-MONTH TOP 3 · CHECKS</div>
<div class="thead"><div></div><div>12-MO</div><div>6-MO<br>TOP 3</div><div>ABOVE<br>200-DAY</div><div>NEAR<br>HIGH</div></div>$rows</div>
<div class="sec"><div class="eyebrow">6-MONTH TOP 3</div><div class="cells">$cells</div>
<div class="legend"><i></i><span>Also in the 12-month top 3</span></div></div>
<div class="foot"><div><b style="color:var(--text)">Plan:</b> hold 1 year · stop-loss 35% below purchase · $$100–$$200 per period</div>
<div>Near high = within 15% of 52-week high. NEW = not in the last report. Price gain only, dividends excluded. Information only, not investment advice.</div></div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline test)")
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--json", default="data.json")
    ap.add_argument("--asof", help="YYYY-MM-DD; default = last completed close")
    ap.add_argument("--seed", type=int, default=7, help="demo mode random seed")
    a = ap.parse_args()

    prev = load_previous(a.json)   # read BEFORE we overwrite it
    if a.demo:
        uni, px = demo_data(a.seed)
    else:
        uni = load_universe()
        px = load_prices(uni["ticker"].tolist())
    res = compute(uni, px, a.asof)

    with open(a.out, "w", encoding="utf-8") as f:
        f.write(render(res, demo=a.demo, prev=prev))
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(dict(asof=str(res["asof"].date()),
                       top12=res["top12"].to_dict("records"),
                       top6=res["top6"].to_dict("records"),
                       buys=[res["top12"].loc[i, "ticker"] for i in res["buys"]]),
                  f, indent=2, default=str)
    buys = [res["top12"].loc[i, "ticker"] for i in res["buys"]]
    print(f"as of {res['asof'].date()} | buys: {buys or 'none'}")
    print(res["top12"][["ticker", "ret12", "ret6", "in6", "above200", "near_high", "passes"]].round(3).to_string(index=False))
    print(res["top6"][["ticker", "ret6", "in12"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
