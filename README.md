# FraudShield

Real-time card fraud detection: a simulator streams transactions, a trained model scores each one in under 50 ms, and analysts review flagged alerts on a live dashboard.

> **Status:** Week 1 — setup, EDA and baseline model. Live demo link, demo GIF and results table land here as the project grows.

## Quick start

```bash
git clone https://github.com/<your-username>/fraudshield.git
cd fraudshield
make setup            # create .venv and install ML dependencies
make data             # download IEEE-CIS data from Kaggle (needs Kaggle CLI + accepted rules)
make train-baseline   # logistic regression baseline, logged to MLflow
make mlflow           # open the MLflow UI at http://localhost:5000
make test             # run the test suite
```

## Repository layout

```
ml/          training, features, evaluation (Python)
api/         FastAPI scoring service           (week 3)
stream/      Redpanda producer + consumer      (week 4)
dashboard/   React + Vite live dashboard       (week 5)
monitoring/  Evidently drift reports           (week 5)
scripts/     data download and helpers
```

## Design decisions so far

- **Time-based split, not random.** Transactions are ordered by `TransactionDT`; train on the earliest 70%, validate on the next 15%, test on the last 15%. A random split leaks future behaviour into training and inflates scores.
- **PR-AUC as the headline metric.** Fraud is a small minority class, so accuracy is meaningless (predicting "not fraud" every time scores ~96%). PR-AUC and recall at a fixed precision reflect what an analyst actually sees.
- **Raw data never enters git.** Only `scripts/download_data.sh` is committed; check the Kaggle dataset rules before publishing anything derived from it.

## License

MIT — see [LICENSE](LICENSE).
