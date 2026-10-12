# MarketMomentum

Twice-monthly S&P 500 momentum report (price gain only, dividends excluded).

- Runs on the 1st and 15th via GitHub Actions (`.github/workflows/report.yml`), or by hand from the Actions tab.
- `momentum.py` ranks S&P 500 stocks by 12-month and 6-month price gain as of the last completed close.
- Checks on the 12-month top 3: in the 6-month top 3, above the 200-day average, within 15% of the 52-week high.
- Buy candidates must pass all three (max 2, one per sector).
- Output: `index.html` (the card, follows light/dark) and `data.json`.

Setup: Settings > Pages > Deploy from a branch > `main` / root. The page is then at
`https://wwjpodcast.github.io/MarketMomentum/`.

Test without internet: `python momentum.py --demo --out demo.html` (synthetic data).

Information only, not investment advice. No account balances are stored here.
