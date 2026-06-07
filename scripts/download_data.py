"""
Downloads the IEEE-CIS Fraud Detection dataset from Kaggle.
Requires: pip install kaggle
          ~/.kaggle/kaggle.json with your API credentials
"""

import subprocess
import sys
from pathlib import Path


def download():
    data_dir = Path("data/ieee_cis")
    data_dir.mkdir(parents=True, exist_ok=True)

    if (data_dir / "train_transaction.csv").exists():
        print("Dataset already downloaded.")
        return

    print("Downloading IEEE-CIS Fraud Detection dataset from Kaggle...")
    print("This is ~364MB — takes a few minutes on a typical connection.\n")

    try:
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "ieee-fraud-detection",
            "-p", str(data_dir)
        ], check=True)
    except FileNotFoundError:
        print("kaggle CLI not found. Install it: pip install kaggle")
        print("Then set up credentials: https://github.com/Kaggle/kaggle-api#api-credentials")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("\nDownload failed. You may need to accept the competition rules at:")
        print("https://www.kaggle.com/c/ieee-fraud-detection/rules")
        sys.exit(1)

    print("\nExtracting...")
    import zipfile
    for f in data_dir.glob("*.zip"):
        with zipfile.ZipFile(f) as z:
            z.extractall(data_dir)
        f.unlink()
        print(f"  Extracted {f.name}")

    print(f"\nDataset ready at {data_dir}/")
    print("Files:")
    for f in data_dir.glob("*.csv"):
        size_mb = f.stat().st_size / 1_048_576
        print(f"  {f.name}: {size_mb:.1f}MB")


if __name__ == "__main__":
    download()