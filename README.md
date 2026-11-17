# Vymera: explained volatility alerts for young investors

**IE University · Fintech · Youth Fintech MVP · Assigned risk: Market Risk**
Team: Alexis, Valentina, Franziska, Teo, Gustavo, Diego

Vymera helps investors aged 18–30 understand *why* their portfolio moves, warns them only when it matters, and lets them act on protection rules they set while calm, always with their approval.

**Live MVP:** https://teodemilecamps.github.io/vymera/

## Try it (2 minutes)
1. Open the live link (works on laptop and phone).
2. Onboarding: choose your time horizon and gut reaction, then **"Enter my own portfolio"**. Type rough euro amounts (e.g. S&P 500 ETF €800, Tesla €600, Bitcoin €400) → **See my risk assessment**.
3. You get a personal **market-risk score (0–100)** with the reasons and tips, plus look-through exposure (hidden concentration inside ETFs, currency share, sensitivity to the market).
4. Use the **demo panel** (right side on laptop, "Demo ▲" button on phone) to trigger market events:
   - *Tech stock drop / Dollar swing*: explained alert splitting the move into market, stock-specific and currency parts, with Hold / Rebalance / Reduce options matched to your plan
   - *Replay April 2025*: real VIX data day by day; the ML model turns "Stormy" on 3 Mar 2025 and your protection rule asks for a one-tap confirmation (undo within 24h)
   - *Hype buy*: 10-second cool-off before buying a trending stock

## Machine-learning model
`ml/vymera_volatility_model.py` trains a logistic regression on CBOE VIX daily data (1990–2015, downloaded automatically on first run) to predict the probability of a high-volatility regime (VIX ≥ 25) within 10 trading days. Test results on unseen 2019–2026 data: AUC 0.91 (naive rule 0.78), 31% of storms flagged before they start, 4.9% false alarms on calm days. The trained weights (`ml/outputs/model_weights.json`) are embedded in the web app, which recomputes the forecast in JavaScript (matches Python to within 0.00005).

```bash
cd ml && pip install -r requirements.txt
python vymera_volatility_model.py   # downloads data, trains, evaluates, exports
python ../mvp/src/build.py          # rebuilds index.html with the new weights
```

## Structure
| Path | What |
|---|---|
| `index.html` | The deployed app (single file, no backend, works offline) |
| `mvp/src/app.template.html` | App source; `build.py` injects model + data |
| `ml/` | Model code, requirements, exported weights and metrics |

Vymera is an information tool: no personal investment advice, and no trades without user confirmation. Broker connections and prices are simulated in this MVP.
