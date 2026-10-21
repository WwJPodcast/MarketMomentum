# MarketMomentum

S&P 500 momentum report, refreshed every weekday (price gain only, dividends excluded). Paydays are every other Friday.

- Runs weekdays just after midnight Central via GitHub Actions (`.github/workflows/report.yml`), or by hand from the Actions tab. Next payday: Fri 2026-10-23.
- `momentum.py` ranks S&P 500 stocks by 12-month and 6-month price gain as of the last completed close.
- Candidates are the 12-month top 3 plus the 6-month top 3 (3 to 6 names).
- Three checks per candidate: in both top 3 lists, above the 200-day average, within 15% of the 52-week high.
- A candidate that fails 2 or more checks is skipped. Buys are the best remaining (3/3 first, then 2/3), max 2, one per sector.
- Market gate: buys are ON while the S&P 500 closes above its 200-day average; it flips only after a 1% close beyond it. While OFF the card shows the names as WATCH and the payday buy is $0.
- Each card also shows how far the price is above its 50-day average (information only).
- NEW = not in the previous payday report (`paydays.json`, saved on payday runs).
- Page 2 (swipe up): a speculative moonshot watchlist from `watchlist.json` (edit the 1-5 `theme` and `bottleneck` ratings and the ticker list). `moonshots.py` scores each name (ratings, revenue growth, cash runway, dilution, size, momentum), ranks the top 15 by score plus days on the list, and keeps the streaks in `moonshot_history.json`. Fundamentals refresh weekly (`fundamentals_cache.json`); prices daily. Tap a column header to sort. If page 2 fails, page 1 still publishes.
- Output: `index.html` (the card, follows light/dark), `data.json`, `paydays.json`, `moonshot_history.json`, `fundamentals_cache.json`.

Pages: Settings > Pages > Deploy from a branch > `main` / root.
Page: `https://wwjpodcast.github.io/MarketMomentum/`

Test without internet: `python momentum.py --demo --out demo.html` (synthetic data).

Information only, not investment advice. No account balances are stored here.
