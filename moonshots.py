"""Page 2 of the card: a speculative watchlist, ranked by a score plus days on the list.

Score (0-100):
  theme fit 15 + bottleneck position 10      (1-5 ratings in watchlist.json, you edit them)
  growth 20        revenue growth YoY (15) + growth speeding up (5)
  survival 15      free cash flow positive, or months of cash left at the current burn
  dilution 10      share count change over the last year (shrinking = full points)
  size 10          mid-size is best, mega caps and tiny caps score lower
  momentum 20      6-month gain vs the rest of the list (12), above 200-day (4), near 52-week high (4)
Rank = score + up to 10 bonus points for the current streak of days on the list (full at 30 days).
Missing data scores 0 for that item. Fundamentals refresh weekly (cached); prices daily.
Information only, not investment advice.
"""
import datetime as dt
import html as _html
import json

import numpy as np
import pandas as pd

TOP_N = 15
STREAK_FULL = 30
STREAK_BONUS = 10
FUND_MAX_AGE_DAYS = 7
NEW_DAYS = 3


# ---------- inputs ----------
def load_watchlist(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["stocks"]


def load_prices(tickers):
    import yfinance as yf
    px = yf.download(tickers, period="15mo", auto_adjust=False, progress=False, threads=True)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    return px.dropna(how="all")


def _row(df, names):
    for n in names:
        if df is not None and n in df.index:
            return df.loc[n].dropna()
    return None


def fetch_fundamentals(ticker):
    """Latest fundamentals for one ticker. Any field can come back None."""
    import yfinance as yf
    out = dict(rev_yoy=None, accel=None, cash=None, debt=None, fcf_ttm=None,
               sh_change=None, mcap=None)
    tk = yf.Ticker(ticker)
    try:
        rev = _row(tk.quarterly_income_stmt, ["Total Revenue", "Operating Revenue"])
        if rev is not None and len(rev) >= 5 and rev.iloc[4] > 0:
            out["rev_yoy"] = float(rev.iloc[0] / rev.iloc[4] - 1)
            if rev.iloc[1] > 0:
                seq = float((rev.iloc[0] / rev.iloc[1]) ** 4 - 1)
                out["accel"] = bool(seq > out["rev_yoy"])
    except Exception:
        pass
    try:
        cf = tk.quarterly_cashflow
        fcf = _row(cf, ["Free Cash Flow"])
        if fcf is None:
            ocf = _row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
            capex = _row(cf, ["Capital Expenditure"])
            if ocf is not None:
                n = min(4, len(ocf))
                out["fcf_ttm"] = float(ocf.iloc[:n].sum() + (capex.iloc[:n].sum() if capex is not None else 0))
        else:
            out["fcf_ttm"] = float(fcf.iloc[:4].sum())
    except Exception:
        pass
    try:
        bs = tk.quarterly_balance_sheet
        cash = _row(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
        debt = _row(bs, ["Total Debt"])
        sh = _row(bs, ["Ordinary Shares Number", "Share Issued"])
        if cash is not None:
            out["cash"] = float(cash.iloc[0])
        if debt is not None:
            out["debt"] = float(debt.iloc[0])
        if sh is not None and len(sh) >= 5 and sh.iloc[4] > 0:
            out["sh_change"] = float(sh.iloc[0] / sh.iloc[4] - 1)
    except Exception:
        pass
    try:
        out["mcap"] = float(tk.fast_info["marketCap"])
    except Exception:
        pass
    return out


def get_fundamentals(tickers, cache_path, today, force=False):
    try:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    fresh = {}
    for t in tickers:
        e = cache.get(t)
        age = (today - dt.date.fromisoformat(e["fetched"])).days if e and "fetched" in e else None
        if not force and e and age is not None and age < FUND_MAX_AGE_DAYS:
            fresh[t] = e
            continue
        try:
            data = fetch_fundamentals(t)
        except Exception:
            data = {}
        if not data or all(v is None for v in data.values()):
            if e:                       # keep the old copy rather than lose it
                fresh[t] = e
                continue
            data = {}
        data["fetched"] = str(today)
        fresh[t] = data
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(fresh, f, indent=1)
    return fresh


def demo_inputs(wl, index, seed=7):
    rng = np.random.default_rng(seed)
    tickers = [w["ticker"] for w in wl]
    n = len(tickers)
    rets = rng.normal(rng.normal(0.0008, 0.0015, n), rng.uniform(0.015, 0.04, n), (len(index), n))
    px = pd.DataFrame(20 * np.exp(np.cumsum(rets, axis=0)), index=index, columns=tickers)
    fund = {}
    for t in tickers:
        fund[t] = dict(rev_yoy=float(rng.uniform(-0.1, 0.9)), accel=bool(rng.random() < 0.5),
                       cash=float(rng.uniform(0.2e9, 6e9)), debt=float(rng.uniform(0, 3e9)),
                       fcf_ttm=float(rng.normal(-1e8, 3e8)), sh_change=float(rng.uniform(-0.02, 0.3)),
                       mcap=float(np.exp(rng.uniform(np.log(3e8), np.log(6e10)))))
    return px, fund


# ---------- scoring ----------
def _lin(x, lo, hi):
    return float(min(max((x - lo) / (hi - lo), 0.0), 1.0))


def price_metrics(s, asof):
    s = s.dropna()
    s = s[s.index <= asof]
    if len(s) < 60:
        return None
    last = s.index[-1]
    c = float(s.iloc[-1])
    m = dict(price=c, ret6=None, above200=False, near_high=False)
    d6 = last - pd.DateOffset(months=6)
    if s.index[0] <= d6:
        m["ret6"] = float(c / s[s.index <= d6].iloc[-1] - 1)
    if len(s) >= 200:
        m["above200"] = bool(c > s.iloc[-200:].mean())
    if len(s) >= 126:
        m["near_high"] = bool(c >= 0.85 * s.iloc[-252:].max())
    return m


def score_one(w, pm, f, mom_pct):
    p = {}
    p["theme"] = 15 * w.get("theme", 0) / 5
    p["bottleneck"] = 10 * w.get("bottleneck", 0) / 5
    g = 0.0
    if f.get("rev_yoy") is not None:
        g += 15 * _lin(f["rev_yoy"], 0.0, 0.6)
    if f.get("accel"):
        g += 5
    p["growth"] = g
    surv = 0.0
    fcf, cash, debt = f.get("fcf_ttm"), f.get("cash"), f.get("debt")
    if fcf is not None and cash is not None:
        if fcf >= 0:
            surv = 15.0
        else:
            months = cash / (-fcf / 12) if fcf < 0 else 99
            surv = 15 * _lin(months, 6, 24)
        if debt is not None and cash > 0 and debt > 2 * cash:
            surv *= 0.5
    p["survival"] = surv
    p["dilution"] = 10 * (1 - _lin(f["sh_change"], 0.0, 0.20)) if f.get("sh_change") is not None else 0.0
    mc = f.get("mcap")
    if mc is None:
        p["size"] = 0.0
    elif mc < 3e8:
        p["size"] = 2.0
    elif mc < 5e9:
        p["size"] = 10.0
    elif mc < 2e10:
        p["size"] = 6.0
    else:
        p["size"] = 3.0
    mo = 0.0
    if pm:
        if mom_pct is not None:
            mo += 12 * mom_pct
        mo += 4 * pm["above200"] + 4 * pm["near_high"]
    p["momentum"] = mo
    return p, float(sum(p.values()))


def rank_list(wl, px, fund, asof, prev_streaks):
    pms = {w["ticker"]: price_metrics(px[w["ticker"]], asof) for w in wl if w["ticker"] in px.columns}
    rets = pd.Series({t: m["ret6"] for t, m in pms.items() if m and m["ret6"] is not None})
    pct = rets.rank(pct=True).to_dict() if len(rets) else {}
    rows, missing = [], []
    for w in wl:
        t = w["ticker"]
        pm = pms.get(t)
        if pm is None:
            missing.append(t)
            continue
        parts, score = score_one(w, pm, fund.get(t, {}), pct.get(t))
        bonus = STREAK_BONUS * min(prev_streaks.get(t, 0), STREAK_FULL) / STREAK_FULL
        rows.append(dict(ticker=t, name=w["name"], segment=w["segment"], score=score,
                         rank_key=score + bonus, price=pm["price"], ret6=pm["ret6"],
                         parts={k: round(v, 1) for k, v in parts.items()}))
    rows.sort(key=lambda r: -r["rank_key"])
    return rows[:TOP_N], missing


# ---------- history ----------
def load_history(path):
    try:
        with open(path, encoding="utf-8") as f:
            h = json.load(f)
        return h if isinstance(h, dict) else {}
    except (OSError, ValueError):
        return {}


def update_history(hist, top, asof):
    """Streak = consecutive report dates on the list. Re-running the same date changes nothing."""
    asof = str(asof)
    if hist.get("asof") == asof:
        return hist
    prev = hist.get("tickers", {})
    cur = {}
    for r in top:
        e = prev.get(r["ticker"])
        cur[r["ticker"]] = dict(streak=(e["streak"] + 1) if e else 1,
                                first=e["first"] if e else asof)
    return dict(asof=asof, tickers=cur)


# ---------- rendering ----------
CSS = """
html{scroll-snap-type:y proximity}
.page{min-height:100vh;min-height:100svh;scroll-snap-align:start}
.page>.wrap{min-height:inherit}
.ms-head,.ms-row{display:grid;grid-template-columns:86px 1fr 32px 74px;gap:6px;align-items:center}
.ms-head{align-items:end;padding-bottom:2px}
.ms-head button{font:9px 'DM Mono',monospace;letter-spacing:.06em;background:none;border:0;padding:6px 0;margin:0;cursor:pointer;color:var(--muted);min-height:32px}
.ms-head button.on{color:var(--blue);font-weight:500}
.ms-row{padding:6px 0;border-top:1px solid var(--line)}
.ms-row:last-child{border-bottom:1px solid var(--line)}
.ms-tk{display:flex;align-items:baseline;gap:7px}.ms-tk .n{font:11px 'DM Mono',monospace;color:var(--muted);width:14px}.ms-tk b{font-size:16px}
.ms-seg{display:inline-block;font:9px 'DM Mono',monospace;letter-spacing:.06em;border:1px solid var(--line);border-radius:4px;padding:2px 5px;white-space:nowrap}
.ms-sc{font:500 13px 'DM Mono',monospace;text-align:right}
.ms-days{display:flex;align-items:center;gap:6px;justify-content:flex-end;width:100%}
.ms-bar{flex:1;height:5px;background:var(--track);border-radius:3px}.ms-bar i{display:block;height:5px;background:var(--blue-bar);border-radius:3px}
.ms-d{font:11px 'DM Mono',monospace;color:var(--muted);width:24px;text-align:right}
"""

JS = """
(function(){
var body=document.getElementById('ms-body');if(!body)return;
var btns=document.querySelectorAll('#ms-head button');
var dflt={rank:1,tk:1,seg:1,score:-1,days:-1};
var cur={k:'rank',d:1};
function val(r,k){return (k==='tk'||k==='seg')?r.dataset[k]:+r.dataset[k];}
function apply(){
 var rows=Array.prototype.slice.call(body.children);
 rows.sort(function(a,b){var x=val(a,cur.k),y=val(b,cur.k);
  var c=(typeof x==='string'?x.localeCompare(y):x-y)*cur.d;
  return c||(+a.dataset.rank-+b.dataset.rank);});
 rows.forEach(function(r){body.appendChild(r);});
 btns.forEach(function(b){var on=b.dataset.k===cur.k;
  b.className=on?'on':'';b.setAttribute('aria-pressed',on?'true':'false');
  b.textContent=b.dataset.l+(on?(cur.d===1?' \\u25B2':' \\u25BC'):'');});
}
btns.forEach(function(b){b.addEventListener('click',function(){
 var k=b.dataset.k;
 if(cur.k===k){cur.d=-cur.d;}else{cur.k=k;cur.d=dflt[k];}
 apply();});});
apply();
})();
"""


def _esc(s):
    return _html.escape(str(s), quote=True)


def _rows_html(top, streaks):
    out = []
    for i, r in enumerate(top, 1):
        d = streaks.get(r["ticker"], {}).get("streak", 1)
        if d <= NEW_DAYS:
            right = '<span class="new">NEW</span>'
        else:
            right = (f'<div class="ms-bar"><i style="width:{min(d, STREAK_FULL) / STREAK_FULL * 100:.0f}%"></i></div>'
                     f'<span class="ms-d">{d}d</span>')
        out.append(
            f'<div class="ms-row" data-rank="{i}" data-tk="{_esc(r["ticker"])}" data-seg="{_esc(r["segment"])}" '
            f'data-score="{r["score"]:.1f}" data-days="{d}">'
            f'<div class="ms-tk"><span class="n">{i}</span><b>{_esc(r["ticker"])}</b></div>'
            f'<div><span class="ms-seg">{_esc(r["segment"])}</span></div>'
            f'<div class="ms-sc">{r["score"]:.0f}</div>'
            f'<div class="ms-days">{right}</div></div>')
    return "".join(out)


def page_html(top, streaks, asof, updated, demo=False):
    def b(k, label):
        return f'<button type="button" data-k="{k}" data-l="{label}">{label}</button>'
    head = (f'<div class="ms-head" id="ms-head">'
            f'<div style="display:flex;gap:10px">{b("rank", "RANK")}{b("tk", "A–Z")}</div>'
            f'<div>{b("seg", "SEGMENT")}</div>'
            f'<div style="text-align:right">{b("score", "SCORE")}</div>'
            f'<div style="text-align:right">{b("days", "DAYS")}</div></div>')
    pill = "Demo data (synthetic)" if demo else "Speculative"
    if not top:
        body = '<div class="none">No watchlist data today.</div>'
    else:
        body = f'<div id="ms-body">{_rows_html(top, streaks)}</div>'
    return f"""<section class="page"><div class="wrap">
<div><div class="top"><div class="eyebrow">PAGE 2 OF 2</div><div class="pill">{pill}</div></div>
<h1>Moonshot watchlist</h1><div class="sub">{_esc(updated)} · prices as of {_esc(asof)} close · $25 per name</div></div>
<div class="row"><div class="lbl">TOP {TOP_N} · TAP A HEADER TO SORT</div><div class="mono muted sm">3–4 names</div></div>
<div>{head}{body}</div>
<div class="foot"><div><b style="color:var(--text)">Rank</b> = score plus days on the list (bar full at {STREAK_FULL} days). Tap a header again to reverse.</div>
<div>Score: theme fit, bottleneck position, growth, cash runway, dilution, size, momentum. Speculative: money you can lose entirely. Separate from the page-1 buys. Information only, not investment advice.</div></div>
</div></section><script>{JS}</script>"""


def fallback_html(updated):
    return (f'<section class="page"><div class="wrap"><div><div class="top"><div class="eyebrow">PAGE 2 OF 2</div>'
            f'<div class="pill">Speculative</div></div><h1>Moonshot watchlist</h1>'
            f'<div class="sub">{_esc(updated)}</div></div>'
            f'<div class="none">Watchlist data was unavailable today. Page 1 is unaffected.</div></div></section>')


def build(asof, today, wl_path, hist_path, cache_path, demo=False, seed=7, index=None):
    """Returns (page2_html, summary_dict). Writes history and cache unless demo."""
    wl = load_watchlist(wl_path)
    asof_ts = pd.Timestamp(asof)
    updated = "Updated " + today.strftime("%a, %b %-d, %Y")
    asof_txt = asof_ts.strftime("%a, %b %-d")
    if demo:
        px, fund = demo_inputs(wl, index, seed)
        rng = np.random.default_rng(seed)
        hist = {}
    else:
        tickers = [w["ticker"] for w in wl]
        px = load_prices(tickers)
        if px.index.tz is not None:
            px.index = px.index.tz_localize(None)
        fund = get_fundamentals(tickers, cache_path, today)
        hist = load_history(hist_path)
    prev_streaks = {t: e["streak"] for t, e in hist.get("tickers", {}).items()}
    top, missing = rank_list(wl, px, fund, asof_ts, prev_streaks)
    if demo:
        streaks = {r["ticker"]: dict(streak=int(rng.integers(1, 31))) for r in top}
        new_hist = hist
    else:
        new_hist = update_history(hist, top, asof_ts.date())
        streaks = new_hist["tickers"]
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(new_hist, f, indent=1)
    summary = dict(asof=str(asof_ts.date()), missing=missing,
                   top=[dict(ticker=r["ticker"], segment=r["segment"], score=round(r["score"], 1),
                             streak=streaks.get(r["ticker"], {}).get("streak"), parts=r["parts"])
                        for r in top])
    return page_html(top, streaks, asof_txt, updated, demo), summary
