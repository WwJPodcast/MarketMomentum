#!/usr/bin/env python3
"""S&P 500 momentum card (price gain only, dividends excluded).

Usage:
  python momentum.py --out index.html --json data.json   # live data (needs internet)
  python momentum.py --demo --out demo.html              # synthetic data, offline test

Rules:
  * Rank S&P 500 stocks by 12-month and 6-month price gain as of the last completed close.
  * Candidates = the 12-month top 3 plus the 6-month top 3 (3 to 6 names).
  * Three checks per candidate (closing prices):
      1. in BOTH top 3 lists
      2. above its 200-day average
      3. within 15% of its 52-week high
  * A candidate failing 2 or more checks is skipped.
  * BUY = best remaining candidates (3/3 first, then 2/3; ties by 12-month gain),
    max 2 picks, at most one per sector.
  * NEW = not in the previous report's lists (read from --json before it is overwritten).
"""
import argparse
import datetime as dt
import io
import json
from string import Template

import numpy as np
import pandas as pd

TOP_N = 3
NEAR_HIGH = 0.85   # within 15% of 52-week high
MIN_SCORE = 2      # need at least 2 of 3 checks
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
    s12, s6 = set(top12["ticker"]), set(top6["ticker"])

    pool = df[df["ticker"].isin(s12 | s6)].copy()
    pool["in12"] = pool["ticker"].isin(s12)
    pool["in6"] = pool["ticker"].isin(s6)
    pool["both"] = pool["in12"] & pool["in6"]
    pool["score"] = pool[["both", "above200", "near_high"]].sum(axis=1).astype(int)
    pool["skip"] = pool["score"] < MIN_SCORE
    pool = pool.sort_values(["skip", "score", "ret12"],
                            ascending=[True, False, False]).reset_index(drop=True)

    buys, seen = [], set()
    for i, r in pool.iterrows():
        if not r["skip"] and r["sector"] not in seen and len(buys) < BUY_MAX:
            buys.append(i)
            seen.add(r["sector"])
    return dict(asof=last, pool=pool, top12=top12, top6=top6, buys=buys)


def load_previous(path):
    """Tickers listed in the previous report (empty set if none)."""
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return set()
    names = set()
    for key in ("pool", "top12", "top6"):
        names |= {r["ticker"] for r in d.get(key, []) if "ticker" in r}
    return names


# ---------- rendering ----------
CHECK = ('<svg width="{s}" height="{s}" viewBox="0 0 16 16" role="img" aria-label="Pass">'
         '<path d="M3 8.5l3.2 3.2L13 4.8" fill="none" style="stroke:var(--green)" '
         'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')
CROSS = ('<svg width="{s}" height="{s}" viewBox="0 0 16 16" role="img" aria-label="Fail">'
         '<path d="M4 4l8 8M12 4l-8 8" fill="none" style="stroke:var(--muted)" '
         'stroke-width="2" stroke-linecap="round"/></svg>')
NEW_CHIP = '<span class="new">NEW</span>'
SKIP_CHIP = '<span class="skip">SKIP</span>'


def icon(ok, size=16):
    return (CHECK if ok else CROSS).format(s=size)


def pct(x):
    return f"{x * 100:+,.1f}%"


def usd(x):
    return f"${x:,.2f}"


def card(rank, r, is_new):
    checks = "".join(
        f'<div class="chk">{icon(ok, 14)}<span>{label}</span></div>'
        for label, ok in [("In both top 3 lists", r.both), ("Above 200-day", r.above200),
                          ("Within 15% of high", r.near_high)])
    chips = (NEW_CHIP if is_new else "") + (f'<span class="part">{r.score}/3</span>' if r.score < 3 else "")
    return f"""<div class="card">
<div class="tags"><span class="tag">BUY · #{rank}</span>{chips}</div>
<div class="big">{r.ticker}</div><div class="muted sm">{r['name']}</div>
<div class="mono muted sm">{usd(r.price)}</div>
<div class="row mt"><span class="lab">12-MO</span><span class="mono g lg">{pct(r.ret12)}</span></div>
<div class="row"><span class="lab">6-MO</span><span class="mono g md">{pct(r.ret6)}</span></div>
<div class="chks">{checks}</div></div>"""


def gain(value, is_leader, size):
    cls = "g b" if is_leader else "m"
    return f'<span class="mono {cls}" style="font-size:{size}px">{pct(value)}</span>'


def render(res, demo=False, prev=frozenset()):
    pool, buys, last = res["pool"], res["buys"], res["asof"]
    new = (set(pool["ticker"]) - set(prev)) if prev else set()
    if buys:
        body = '<div class="cards">' + "".join(
            card(n + 1, pool.loc[i], pool.loc[i, "ticker"] in new) for n, i in enumerate(buys)) + "</div>"
        head = "BUY CANDIDATES"
    else:
        body = '<div class="none">No buy this period. No leader passed at least 2 of 3 checks.</div>'
        head = "NO BUY THIS PERIOD"
    rows = ""
    for _, r in pool.iterrows():
        chips = (NEW_CHIP if r.ticker in new else "") + (SKIP_CHIP if r.skip else "")
        cls = "trow dim" if r.skip else "trow"
        rows += (f'<div class="{cls}"><div class="tk"><b>{r.ticker}</b>{chips}</div>'
                 f'{gain(r.ret12, r.in12, 11)}{gain(r.ret6, r.in6, 11)}'
                 f'<div class="c">{icon(r.both)}</div><div class="c">{icon(r.above200)}</div>'
                 f'<div class="c">{icon(r.near_high)}</div></div>')
    pill = "Demo data (synthetic)" if demo else "Live data"
    return Template(PAGE).substitute(
        pill=pill, report=dt.date.today().strftime("%a, %b %-d, %Y"),
        asof=last.strftime("%a, %b %-d"), head=head, body=body, rows=rows)


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
.wrap{max-width:390px;margin:0 auto;padding:22px 18px 16px;display:flex;flex-direction:column;gap:14px}
.mono{font-family:'DM Mono',monospace}.muted{color:var(--muted)}.g{color:var(--green)}.m{color:var(--muted)}.b{font-weight:500}.sm{font-size:12px}.md{font-size:15px}.lg{font-size:19px}
.eyebrow{font:11px 'DM Mono',monospace;letter-spacing:.14em;color:var(--muted)}
.top{display:flex;justify-content:space-between;align-items:center}
.pill{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:2px 8px}
h1{margin:6px 0;font-size:28px;line-height:1.1}.sub{font-size:13px;color:var(--muted)}
.sec{display:flex;flex-direction:column;gap:10px}.lbl{font:11px 'DM Mono',monospace;letter-spacing:.14em;color:var(--blue)}
.cards{display:flex;gap:10px}.card{flex:1;min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px;display:flex;flex-direction:column;gap:4px}
.row{display:flex;justify-content:space-between;align-items:baseline}.mt{margin-top:4px}.tags{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
.tag{font:500 10px 'DM Mono',monospace;letter-spacing:.1em;background:var(--strip);color:#fff;border-radius:4px;padding:3px 7px;white-space:nowrap}
.new{font:10px 'DM Mono',monospace;letter-spacing:.08em;background:var(--text);color:var(--bg);border-radius:4px;padding:2px 5px;white-space:nowrap}
.part{font:10px 'DM Mono',monospace;border:1px solid var(--muted);color:var(--text);border-radius:4px;padding:1px 5px;white-space:nowrap}
.skip{font:9px 'DM Mono',monospace;letter-spacing:.06em;border:1px solid var(--muted);color:var(--muted);border-radius:4px;padding:1px 4px;white-space:nowrap}
.big{font:600 36px/1 'Fraunces',serif;margin-top:4px}.lab{font:10px 'DM Mono',monospace;letter-spacing:.1em;color:var(--muted)}
.chks{display:flex;flex-direction:column;gap:4px;margin-top:6px;padding-top:7px;border-top:1px solid var(--line)}
.chk{display:flex;align-items:center;gap:7px;font:10px 'DM Mono',monospace}.none{border:1px dashed var(--line);border-radius:10px;padding:16px;color:var(--muted);font-size:14px}
.thead,.trow{display:grid;grid-template-columns:92px 64px 58px 1fr 1fr 1fr;gap:4px;align-items:center}
.thead div{font:9px/1.3 'DM Mono',monospace;color:var(--muted);text-align:center}.thead div:nth-child(2),.thead div:nth-child(3){text-align:left}
.trow{padding:7px 0;border-top:1px solid var(--line)}.trow:last-child{border-bottom:1px solid var(--line)}
.tk{display:flex;align-items:center;gap:5px}.tk b{font-size:15px}.trow.dim .tk b{color:var(--muted)}.c{display:flex;justify-content:center}
.legend{font-size:11px;line-height:1.45;color:var(--muted);margin-top:6px}
.foot{border-top:1px solid var(--line);padding-top:10px;font-size:11px;line-height:1.45;color:var(--muted);display:flex;flex-direction:column;gap:5px}
</style></head><body><div class="wrap">
<div><div class="top"><div class="eyebrow">S&amp;P 500 MOMENTUM</div><div class="pill">$pill</div></div>
<h1>Momentum leaders</h1><div class="sub">$report · prices as of $asof close</div></div>
<div class="sec"><div class="row"><div class="lbl">$head</div><div class="mono muted sm">$$100–$$200 total</div></div>$body</div>
<div><div class="eyebrow" style="margin-bottom:6px">12- AND 6-MONTH LEADERS · CHECKS</div>
<div class="thead"><div></div><div>12-MO</div><div>6-MO</div><div>IN BOTH<br>LISTS</div><div>ABOVE<br>200-DAY</div><div>NEAR<br>HIGH</div></div>$rows
<div class="legend">Green gain = in that top 3. SKIP = fails 2 or more checks.</div></div>
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
    buys = [res["pool"].loc[i, "ticker"] for i in res["buys"]]
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(dict(asof=str(res["asof"].date()),
                       pool=res["pool"].to_dict("records"),
                       top12=res["top12"].to_dict("records"),
                       top6=res["top6"].to_dict("records"),
                       buys=buys),
                  f, indent=2, default=str)
    print(f"as of {res['asof'].date()} | buys: {buys or 'none'}")
    cols = ["ticker", "ret12", "ret6", "both", "above200", "near_high", "score", "skip"]
    print(res["pool"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
