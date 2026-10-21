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
  * MARKET GATE: buys are ON while the S&P 500 closes above its 200-day average. The gate
    flips only after a close more than 1% beyond the average (so it does not flicker).
    While OFF the card shows the same names as WATCH and the payday buy is $0.
  * "VS 50-DAY" = how far each price is above its 50-day average. Information only.
  * NEW = not in the previous payday report (paydays.json). Payday = every 14 days from
    PAY_ANCHOR; the payday run saves that day's list.
"""
import argparse
import datetime as dt
import io
import json
from string import Template
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TOP_N = 3
NEAR_HIGH = 0.85   # within 15% of 52-week high
MIN_SCORE = 2      # need at least 2 of 3 checks
BUY_MAX = 2
GATE_BAND = 0.01   # S&P 500 must close 1% beyond its 200-day average to flip the gate
PAY_ANCHOR = dt.date(2026, 6, 5)   # a payday Friday; paydays repeat every 14 days


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


def load_market():
    import yfinance as yf
    s = yf.download("^GSPC", period="15mo", auto_adjust=False, progress=False)["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.dropna()


def demo_market(index, up=True, seed=11):
    rng = np.random.default_rng(seed)
    drift = np.full(len(index), 0.0005)
    if not up:
        drift[-70:] = -0.004   # sell-off: ends well below the 200-day average
    return pd.Series(4000 * np.exp(np.cumsum(rng.normal(drift, 0.004))), index=index)


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
            stretch50=float(c / s.iloc[-50:].mean() - 1),
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


def market_gate(mpx, last, prev_on=None):
    """S&P 500 vs its 200-day average, with a 1% buffer once a state is set."""
    m = mpx.sort_index()
    if m.index.tz is not None:
        m.index = m.index.tz_localize(None)
    m = m[m.index <= last]
    if len(m) < 200:
        raise SystemExit("not enough S&P 500 history for the 200-day average")
    ma, c = float(m.iloc[-200:].mean()), float(m.iloc[-1])
    diff = c / ma - 1
    if prev_on is None:
        on = diff > 0
    elif prev_on:
        on = diff > -GATE_BAND
    else:
        on = diff > GATE_BAND
    return dict(on=bool(on), close=c, ma200=ma, diff=float(diff),
                asof=str(m.index[-1].date()), band=GATE_BAND)


def load_prev_gate(path):
    try:
        with open(path, encoding="utf-8") as f:
            on = json.load(f).get("gate", {}).get("on")
    except (OSError, ValueError):
        return None
    return on if isinstance(on, bool) else None


def local_today():
    return dt.datetime.now(ZoneInfo("America/Chicago")).date()


def is_payday(d):
    return (d - PAY_ANCHOR).days % 14 == 0


def load_paydays(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def baseline(paydays, today):
    """Tickers from the latest saved payday report dated before today (empty if none)."""
    older = [e for e in paydays if e.get("date", "") < str(today)]
    return set(max(older, key=lambda e: e["date"])["tickers"]) if older else set()


def save_paydays(path, paydays, today, tickers):
    """On a payday, record that day's list (re-runs the same day replace it)."""
    if not is_payday(today):
        return
    kept = [e for e in paydays if e.get("date") != str(today)]
    kept.append(dict(date=str(today), tickers=sorted(tickers)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(kept, key=lambda e: e["date"])[-6:], f, indent=2)


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


def gate_html(g):
    side = "above" if g["diff"] >= 0 else "below"
    note = ""
    if g["on"] and g["diff"] < 0:
        note = " (inside the 1% buffer, still on)"
    elif not g["on"] and g["diff"] > 0:
        note = " (not yet 1% over, still off)"
    amount = f'<span class="mono">{abs(g["diff"]) * 100:.1f}%</span>'
    if g["on"]:
        icon_svg = ('<svg width="20" height="20" viewBox="0 0 16 16" role="img" aria-label="Buys on" '
                    'style="flex:none"><circle cx="8" cy="8" r="7" fill="none" style="stroke:var(--green)" '
                    'stroke-width="1.6"/><path d="M4.6 8.4l2.4 2.4 4.4-4.8" fill="none" '
                    'style="stroke:var(--green)" stroke-width="1.8" stroke-linecap="round" '
                    'stroke-linejoin="round"/></svg>')
        title, tail = "MARKET GATE · BUYS ON", ""
    else:
        icon_svg = ('<svg width="20" height="20" viewBox="0 0 16 16" role="img" aria-label="Buys off" '
                    'style="flex:none"><path d="M5.2 1.6h5.6l3.6 3.6v5.6l-3.6 3.6H5.2l-3.6-3.6V5.2z" '
                    'fill="none" style="stroke:var(--off)" stroke-width="1.6" stroke-linejoin="round"/>'
                    '<path d="M5 8h6" fill="none" style="stroke:var(--off)" stroke-width="2" '
                    'stroke-linecap="round"/></svg>')
        title, tail = "MARKET GATE · BUYS OFF", " Hold this payday's cash."
    return (f'<div class="gate {"on" if g["on"] else "off"}">{icon_svg}<div>'
            f'<div class="gl">{title}</div>'
            f'<div class="gs">S&amp;P 500 is {amount} {side} its 200-day average{note}.{tail}</div>'
            f'</div></div>')


def card(rank, r, is_new, watch=False):
    checks = "".join(
        f'<div class="chk">{icon(ok, 14)}<span>{label}</span></div>'
        for label, ok in [("In both top 3 lists", r.both), ("Above 200-day", r.above200),
                          ("Within 15% of high", r.near_high)])
    chips = (NEW_CHIP if is_new else "") + (f'<span class="part">{r.score}/3</span>' if r.score < 3 else "")
    return f"""<div class="card">
<div class="tags"><span class="tag{" watch" if watch else ""}">{"WATCH" if watch else "BUY"} · #{rank}</span>{chips}</div>
<div class="big">{r.ticker}</div><div class="muted sm">{r['name']}</div>
<div class="mono muted sm">{usd(r.price)}</div>
<div class="row mt"><span class="lab">12-MO</span><span class="mono g lg">{pct(r.ret12)}</span></div>
<div class="row"><span class="lab">6-MO</span><span class="mono g md">{pct(r.ret6)}</span></div>
<div class="row"><span class="lab">VS 50-DAY</span><span class="mono muted" style="font-size:13px">{r.stretch50 * 100:+,.0f}%</span></div>
<div class="chks">{checks}</div></div>"""


def gain(value, is_leader, size):
    cls = "g b" if is_leader else "m"
    return f'<span class="mono {cls}" style="font-size:{size}px">{pct(value)}</span>'


def render(res, gate, demo=False, prev=frozenset(), page2="", css2=""):
    pool, buys, last = res["pool"], res["buys"], res["asof"]
    new = (set(pool["ticker"]) - set(prev)) if prev else set()
    watch = not gate["on"]
    total = "$0 this payday" if watch else "$100–$200 total"
    if buys:
        body = '<div class="cards">' + "".join(
            card(n + 1, pool.loc[i], pool.loc[i, "ticker"] in new, watch)
            for n, i in enumerate(buys)) + "</div>"
        head = "WATCH · GATE OFF" if watch else "BUY CANDIDATES"
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
    pill = "Demo data · swipe up ↑" if demo else "Page 1 of 2 · swipe up ↑"
    return Template(PAGE).substitute(
        pill=pill, report=dt.date.today().strftime("%a, %b %-d, %Y"),
        asof=last.strftime("%a, %b %-d"), head=head, body=body, rows=rows,
        gate=gate_html(gate), total=total, page2=page2, css2=css2)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Momentum leaders</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&family=Fraunces:wght@600&display=swap">
<style>
:root{--bg:#F4F1EA;--panel:#fff;--track:#E2DED2;--text:#1B1F2A;--muted:#5A5F6B;--line:#D9D4C7;--blue:#1F3FA6;--blue-bar:#1F3FA6;--strip:#1F3FA6;--green:#0B6B3A;--bar-muted:#8D93A0;--on-bg:#E1F1E7;--on-line:#9CCBB0;--off-bg:#FBE4E1;--off-line:#D98F87;--off:#A3231A}
@media(prefers-color-scheme:dark){:root{--bg:#14171C;--panel:#1D2128;--track:#232830;--text:#ECEAE4;--muted:#A3A8B2;--line:#3A404B;--blue:#8FA8FF;--blue-bar:#6B8CFF;--strip:#3556D4;--green:#4CC585;--bar-muted:#5A6272;--on-bg:#173025;--on-line:#2F6B4D;--off-bg:#35191B;--off-line:#8C4545;--off:#FF8F85}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif}
.wrap{max-width:390px;margin:0 auto;padding:16px 18px 12px;display:flex;flex-direction:column;gap:11px}
.mono{font-family:'DM Mono',monospace}.muted{color:var(--muted)}.g{color:var(--green)}.m{color:var(--muted)}.b{font-weight:500}.sm{font-size:12px}.md{font-size:15px}.lg{font-size:19px}
.eyebrow{font:11px 'DM Mono',monospace;letter-spacing:.14em;color:var(--muted)}
.top{display:flex;justify-content:space-between;align-items:center}
.pill{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:999px;padding:2px 8px}
h1{margin:4px 0;font-size:28px;line-height:1.1}.sub{font-size:13px;color:var(--muted)}
.sec{display:flex;flex-direction:column;gap:10px}.lbl{font:11px 'DM Mono',monospace;letter-spacing:.14em;color:var(--blue)}
.cards{display:flex;gap:10px}.card{flex:1;min-width:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:12px;display:flex;flex-direction:column;gap:4px}
.row{display:flex;justify-content:space-between;align-items:baseline}.mt{margin-top:4px}.tags{display:flex;gap:5px;align-items:center;flex-wrap:wrap}
.tag{font:500 10px 'DM Mono',monospace;letter-spacing:.1em;background:var(--strip);color:#fff;border-radius:4px;padding:3px 7px;white-space:nowrap}
.new{font:10px 'DM Mono',monospace;letter-spacing:.08em;background:var(--text);color:var(--bg);border-radius:4px;padding:2px 5px;white-space:nowrap}
.part{font:10px 'DM Mono',monospace;border:1px solid var(--muted);color:var(--text);border-radius:4px;padding:1px 5px;white-space:nowrap}
.skip{font:9px 'DM Mono',monospace;letter-spacing:.06em;border:1px solid var(--muted);color:var(--muted);border-radius:4px;padding:1px 4px;white-space:nowrap}
.tag.watch{background:var(--track);color:var(--text)}
.gate{display:flex;gap:10px;align-items:center;border:1px solid;border-radius:12px;padding:10px 12px}
.gate.on{background:var(--on-bg);border-color:var(--on-line)}.gate.off{background:var(--off-bg);border-color:var(--off-line)}
.gl{font:500 11px 'DM Mono',monospace;letter-spacing:.1em}.gate.on .gl{color:var(--green)}.gate.off .gl{color:var(--off)}
.gs{font-size:12px;line-height:1.35;margin-top:2px}
.big{font:600 36px/1 'Fraunces',serif;margin-top:4px}.lab{font:10px 'DM Mono',monospace;letter-spacing:.1em;color:var(--muted)}
.chks{display:flex;flex-direction:column;gap:4px;margin-top:6px;padding-top:7px;border-top:1px solid var(--line)}
.chk{display:flex;align-items:center;gap:7px;font:10px 'DM Mono',monospace}.none{border:1px dashed var(--line);border-radius:10px;padding:16px;color:var(--muted);font-size:14px}
.thead,.trow{display:grid;grid-template-columns:104px 62px 54px 1fr 1fr 1fr;gap:4px;align-items:center}
.thead div{font:9px/1.3 'DM Mono',monospace;color:var(--muted);text-align:center}.thead div:nth-child(2),.thead div:nth-child(3){text-align:left}
.trow{padding:5px 0;border-top:1px solid var(--line)}.trow:last-child{border-bottom:1px solid var(--line)}
.tk{display:flex;align-items:center;gap:5px}.tk b{font-size:15px}.trow.dim .tk b{color:var(--muted)}.c{display:flex;justify-content:center}
.legend{font-size:11px;line-height:1.45;color:var(--muted);margin-top:6px}
.foot{margin-top:auto;border-top:1px solid var(--line);padding-top:10px;font-size:11px;line-height:1.45;color:var(--muted);display:flex;flex-direction:column;gap:5px}
$css2
</style></head><body><section class="page"><div class="wrap">
<div><div class="top"><div class="eyebrow">S&amp;P 500 MOMENTUM</div><div class="pill">$pill</div></div>
<h1>Momentum leaders</h1><div class="sub">$report · prices as of $asof close</div></div>
$gate
<div class="sec"><div class="row"><div class="lbl">$head</div><div class="mono muted sm">$total</div></div>$body</div>
<div><div class="eyebrow" style="margin-bottom:6px">12- AND 6-MONTH LEADERS · CHECKS</div>
<div class="thead"><div></div><div>12-MO</div><div>6-MO</div><div>IN BOTH<br>LISTS</div><div>ABOVE<br>200-DAY</div><div>NEAR<br>HIGH</div></div>$rows
<div class="legend">Green gain = in that top 3. SKIP = fails 2 or more checks.</div></div>
<div class="foot"><div><b style="color:var(--text)">Plan:</b> hold 1 year · stop-loss 35% below purchase · $$100–$$200 per payday</div>
<div>Buys pause while the S&amp;P 500 is below its 200-day average. Near high = within 15% of 52-week high. Vs 50-day is information only. NEW = not in the last payday report. Price gain only. Not investment advice.</div></div>
</div></section>$page2</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline test)")
    ap.add_argument("--out", default="index.html")
    ap.add_argument("--json", default="data.json")
    ap.add_argument("--asof", help="YYYY-MM-DD; default = last completed close")
    ap.add_argument("--seed", type=int, default=7, help="demo mode random seed")
    ap.add_argument("--paydays", default="paydays.json", help="saved payday lists (for NEW)")
    ap.add_argument("--moon-watchlist", default="watchlist.json")
    ap.add_argument("--moon-history", default="moonshot_history.json")
    ap.add_argument("--moon-cache", default="fundamentals_cache.json")
    ap.add_argument("--demo-gate", choices=["on", "off"], default="on",
                    help="demo mode: simulate the S&P 500 above/below its 200-day average")
    a = ap.parse_args()

    today = local_today()
    prev_gate = load_prev_gate(a.json)   # read BEFORE we overwrite data.json
    paydays = load_paydays(a.paydays)
    prev = baseline(paydays, today)
    if a.demo:
        uni, px = demo_data(a.seed)
    else:
        uni = load_universe()
        px = load_prices(uni["ticker"].tolist())
    res = compute(uni, px, a.asof)
    mpx = demo_market(px.index, a.demo_gate == "on") if a.demo else load_market()
    gate = market_gate(mpx, res["asof"], prev_gate)

    # page 2 (moonshot watchlist) must never break page 1
    moon = None
    try:
        import moonshots
        page2, moon = moonshots.build(res["asof"], today, a.moon_watchlist, a.moon_history,
                                      a.moon_cache, demo=a.demo, seed=a.seed, index=px.index)
        css2 = moonshots.CSS
        if moon["missing"]:
            print("watchlist tickers with no price data:", ", ".join(moon["missing"]))
    except Exception as e:
        print(f"page 2 failed, publishing page 1 only: {e!r}")
        try:
            import moonshots
            page2, css2 = moonshots.fallback_html("Updated " + today.strftime("%a, %b %-d, %Y")), moonshots.CSS
        except Exception:
            page2, css2 = "", ""
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(render(res, gate, demo=a.demo, prev=prev, page2=page2, css2=css2))
    if not a.demo:
        save_paydays(a.paydays, paydays, today, set(res["pool"]["ticker"]))
    buys = [res["pool"].loc[i, "ticker"] for i in res["buys"]]
    print(f"market gate: {'ON' if gate['on'] else 'OFF'} "
          f"(S&P 500 {gate['diff'] * 100:+.1f}% vs 200-day)")
    with open(a.json, "w", encoding="utf-8") as f:
        json.dump(dict(asof=str(res["asof"].date()), gate=gate,
                       pool=res["pool"].to_dict("records"),
                       top12=res["top12"].to_dict("records"),
                       top6=res["top6"].to_dict("records"),
                       buys=buys, moonshots=moon),
                  f, indent=2, default=str)
    print(f"as of {res['asof'].date()} | buys: {buys or 'none'}")
    cols = ["ticker", "ret12", "ret6", "both", "above200", "near_high", "score", "skip"]
    print(res["pool"][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
