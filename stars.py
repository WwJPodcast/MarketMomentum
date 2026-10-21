"""Page 2 of the card: Star Potential.

Idea (Richard Koch, The Star Principle): the best businesses are the leader of a niche
that is growing 10%+ a year. We pick the segments first (stars.json, researched and reviewed
monthly/quarterly), then the leading stocks inside each one. The 6 segments are ranked; the
top 3 show 3 stocks each and the bottom 3 show 2 each (15 names).

This script never changes the segments or stocks. Each day it only refreshes numbers:
price, 6-month gain, vs 50-day average, drawdown from the 52-week high (alert at -35%),
and a sanity row per segment (median 6-month gain vs the S&P 500, count above 200-day).
Information only, not investment advice.
"""
import html as _html
import json

import numpy as np
import pandas as pd

ALERT_DD = 0.35

CSS = """
html{scroll-snap-type:y proximity}
.page{min-height:100vh;min-height:100svh;scroll-snap-align:start}
.page>.wrap{min-height:inherit}
.st{display:flex;flex-direction:column;gap:5px}
.sg{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:5px 9px 3px}
.sg-h{display:flex;justify-content:space-between;align-items:baseline;gap:6px}
.sg-n{font:500 11px 'DM Mono',monospace;letter-spacing:.06em;color:var(--blue);white-space:nowrap}
.sg-g{font:10px 'DM Mono',monospace;color:var(--muted);white-space:nowrap}
.sg-s{font:9px/1.3 'DM Mono',monospace;color:var(--muted);padding:1px 0 3px}
.sr{display:grid;grid-template-columns:62px 1fr 54px 42px 42px;gap:4px;align-items:center;padding:2px 0;border-top:1px solid var(--line)}
.sr b{font-size:14px}
.sr .rl{font:9px 'DM Mono',monospace;letter-spacing:.04em;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sr .px{font:12px 'DM Mono',monospace;text-align:right}
.sr .rt{font:11px 'DM Mono',monospace;text-align:right}
.sr .rt.p{color:var(--green)}.sr .rt.n{color:var(--off)}.sr .rt.z{color:var(--muted)}
.sr .al{font:500 9px 'DM Mono',monospace;color:var(--off);border:1px solid var(--off);border-radius:4px;padding:0 4px}
.sc{display:grid;grid-template-columns:62px 1fr 54px 42px 42px;gap:4px;font:8px 'DM Mono',monospace;letter-spacing:.06em;color:var(--muted);padding:0 0 2px}
.sc div:nth-child(n+3){text-align:right}
"""


def _esc(s):
    return _html.escape(str(s), quote=True)


def _usd(x):
    return f"${x:,.0f}" if x >= 1000 else f"${x:,.2f}"


def load_stars(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    segs = sorted(d["segments"], key=lambda s: s["rank"])
    return segs, d


def load_prices(tickers):
    import yfinance as yf
    px = yf.download(tickers, period="15mo", auto_adjust=False, progress=False, threads=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    return px.dropna(how="all")


def demo_prices(tickers, index, seed=7):
    rng = np.random.default_rng(seed)
    n = len(tickers)
    rets = rng.normal(rng.normal(0.0008, 0.0015, n), rng.uniform(0.015, 0.035, n), (len(index), n))
    px = pd.DataFrame(20 * np.exp(np.cumsum(rets, axis=0)), index=index, columns=tickers)
    # one name with a short history (like a recent IPO) and one deep drawdown, to exercise the page
    if n > 6:
        px.iloc[:-90, 6] = np.nan
    if n > 3:
        px.iloc[-60:, 3] = px.iloc[-61, 3] * np.linspace(1, 0.5, 60)
    return px


def metrics(s, asof):
    s = s.dropna()
    s = s[s.index <= asof]
    if len(s) < 2:
        return None
    last = s.index[-1]
    c = float(s.iloc[-1])
    m = dict(price=c, ret6=None, vs50=None, above200=None, dd=None, bars=len(s))
    d6 = last - pd.DateOffset(months=6)
    if s.index[0] <= d6:
        m["ret6"] = float(c / s[s.index <= d6].iloc[-1] - 1)
    if len(s) >= 50:
        m["vs50"] = float(c / s.iloc[-50:].mean() - 1)
    if len(s) >= 200:
        m["above200"] = bool(c > s.iloc[-200:].mean())
    m["dd"] = float(c / s.iloc[-252:].max() - 1)
    return m


def _pct(x):
    return "n/a" if x is None else f"{x * 100:+.0f}%"


def _cls(x):
    return "z" if x is None else ("p" if x >= 0 else "n")


def compute(segs, px, mpx, asof):
    sp6 = None
    if mpx is not None:
        sp = mpx.dropna()
        sp = sp[sp.index <= asof]
        d6 = sp.index[-1] - pd.DateOffset(months=6)
        if len(sp) and sp.index[0] <= d6:
            sp6 = float(sp.iloc[-1] / sp[sp.index <= d6].iloc[-1] - 1)
    out, missing = [], []
    for sg in segs:
        rows = []
        for st in sg["stocks"]:
            t = st["ticker"]
            m = metrics(px[t], asof) if t in px.columns else None
            if m is None:
                missing.append(t)
            rows.append(dict(ticker=t, role=st.get("role", ""), m=m))
        r6 = [r["m"]["ret6"] for r in rows if r["m"] and r["m"]["ret6"] is not None]
        a200 = [r["m"]["above200"] for r in rows if r["m"] and r["m"]["above200"] is not None]
        out.append(dict(seg=sg, rows=rows,
                        med6=float(np.median(r6)) if r6 else None,
                        above=(sum(a200), len(a200))))
    return out, sp6, missing


def _seg_html(b, sp6):
    sg = b["seg"]
    med = "n/a" if b["med6"] is None else f"{b['med6'] * 100:+.0f}%"
    spt = "" if sp6 is None else f" vs S&P {sp6 * 100:+.0f}%"
    ab = f"{b['above'][0]}/{b['above'][1]} above 200-day" if b["above"][1] else "200-day n/a"
    rows = []
    for r in b["rows"]:
        m = r["m"]
        if m is None:
            rows.append(f'<div class="sr"><b>{_esc(r["ticker"])}</b><span class="rl">NO DATA</span>'
                        f'<span class="px">—</span><span class="rt z">—</span><span class="rt z">—</span></div>')
            continue
        alert = m["dd"] is not None and m["dd"] <= -ALERT_DD
        mid = (f'<span><span class="al">{m["dd"] * 100:.0f}% FROM HIGH</span></span>' if alert
               else f'<span class="rl">{_esc(r["role"])}</span>')
        rows.append(f'<div class="sr"><b>{_esc(r["ticker"])}</b>{mid}'
                    f'<span class="px">{_usd(m["price"])}</span>'
                    f'<span class="rt {_cls(m["ret6"])}">{_pct(m["ret6"])}</span>'
                    f'<span class="rt {_cls(m["vs50"])}">{_pct(m["vs50"])}</span></div>')
    return (f'<div class="sg"><div class="sg-h"><span class="sg-n">{sg["rank"]} · {_esc(sg["name"])}</span>'
            f'<span class="sg-g">{_esc(sg["growth"])}</span></div>'
            f'<div class="sg-s">Median 6-mo {med}{spt} · {ab}</div>{"".join(rows)}</div>')


def page_html(blocks, sp6, asof, updated, reviewed, demo=False):
    pill = "Demo data (synthetic)" if demo else "Speculative"
    head = ('<div class="sc"><div></div><div></div><div>PRICE</div><div>6-MO</div><div>VS 50D</div></div>')
    body = "".join(_seg_html(b, sp6) for b in blocks)
    return f"""<section class="page"><div class="wrap">
<div><div class="top"><div class="eyebrow">PAGE 2 OF 2</div><div class="pill">{pill}</div></div>
<h1>Star Potential</h1><div class="sub">Prices as of {_esc(asof)} close · leaders of fast-growing niches</div></div>
<div class="st">{head}{body}</div>
<div class="foot"><div><b style="color:var(--text)">Plan:</b> small equal positions, hold 1–2 years. Segments ranked by growth and leader strength, reviewed {_esc(reviewed)}. Red flag = 35% below 52-week high.</div>
<div>Idea from Richard Koch, The Star Principle. Role labels are a judgment, not a fact. Speculative. Price gain only. Not investment advice.</div></div>
</div></section>"""


def fallback_html(updated):
    return (f'<section class="page"><div class="wrap"><div><div class="top"><div class="eyebrow">PAGE 2 OF 2</div>'
            f'<div class="pill">Speculative</div></div><h1>Star Potential</h1>'
            f'<div class="sub">{_esc(updated)}</div></div>'
            f'<div class="none">Star data was unavailable today. Page 1 is unaffected.</div></div></section>')


def build(asof, today, stars_path, demo=False, seed=7, index=None, mpx=None):
    """Returns (page2_html, summary_dict)."""
    segs, meta = load_stars(stars_path)
    tickers = [st["ticker"] for sg in segs for st in sg["stocks"]]
    asof_ts = pd.Timestamp(asof)
    if demo:
        px = demo_prices(tickers, index, seed)
    else:
        px = load_prices(tickers)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
    blocks, sp6, missing = compute(segs, px, mpx, asof_ts)
    updated = "Updated " + today.strftime("%a, %b %-d, %Y")
    html = page_html(blocks, sp6, asof_ts.strftime("%a, %b %-d"), updated,
                     meta.get("reviewed", "?"), demo)
    summary = dict(asof=str(asof_ts.date()), reviewed=meta.get("reviewed"), missing=missing, sp6=sp6,
                   segments=[dict(rank=b["seg"]["rank"], name=b["seg"]["name"], med6=b["med6"],
                                  above200=b["above"],
                                  stocks=[dict(ticker=r["ticker"], role=r["role"], **(r["m"] or {}))
                                          for r in b["rows"]])
                             for b in blocks],
                   alerts=[r["ticker"] for b in blocks for r in b["rows"]
                           if r["m"] and r["m"]["dd"] is not None and r["m"]["dd"] <= -ALERT_DD])
    return html, summary
