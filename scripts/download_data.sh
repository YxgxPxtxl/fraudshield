#!/usr/bin/env bash
# Download the IEEE-CIS Fraud Detection competition data into data/raw.
#
# Before running:
#   1. pip install kaggle   (included in requirements-dev.txt)
#   2. Create an API token at https://www.kaggle.com/settings -> API -> Create New Token.
#      New tokens (KGAT_...): save to ~/.kaggle/access_token (chmod 600) or export KAGGLE_API_TOKEN.
#      Legacy kaggle.json: save to ~/.kaggle/kaggle.json (chmod 600).
#   3. Accept the competition rules at
#      https://www.kaggle.com/competitions/ieee-fraud-detection/rules
#      (the download returns 403 until you do).
#
# Optional backup dataset (smaller, ~0.17% fraud):
#   bash scripts/download_data.sh --backup
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/raw}"
mkdir -p "$DATA_DIR"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Run: pip install kaggle" >&2
  exit 1
fi

if [[ "${1:-}" == "--backup" ]]; then
  echo "Downloading ULB Credit Card Fraud dataset into $DATA_DIR/creditcard ..."
  kaggle datasets download -d mlg-ulb/creditcardfraud -p "$DATA_DIR/creditcard" --unzip
  echo "Done."
  exit 0
fi

if [[ -f "$DATA_DIR/train_transaction.csv" ]]; then
  echo "IEEE-CIS data already present in $DATA_DIR — skipping download."
  exit 0
fi

echo "Downloading IEEE-CIS Fraud Detection data into $DATA_DIR ..."
kaggle competitions download -c ieee-fraud-detection -p "$DATA_DIR"
unzip -o -q "$DATA_DIR/ieee-fraud-detection.zip" -d "$DATA_DIR"
rm -f "$DATA_DIR/ieee-fraud-detection.zip"

echo "Done. Files:"
ls -lh "$DATA_DIR"
