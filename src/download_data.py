"""Download the raw Telco Customer Churn CSV and verify it is the exact file this project was built on."""
import hashlib
import sys
import urllib.request

from src import config

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
# SHA-256 of the file used to build every number in this project.
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def sha256_of(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not config.RAW_DATA_PATH.exists():
        print(f"Downloading {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, config.RAW_DATA_PATH)

    actual = sha256_of(config.RAW_DATA_PATH)
    if actual != EXPECTED_SHA256:
        sys.exit(
            "Checksum mismatch: the raw file differs from the one this project was built on.\n"
            f"  expected {EXPECTED_SHA256}\n  actual   {actual}\n"
            "Results will not match the README until the file matches."
        )
    print("Raw data present and checksum verified.")


if __name__ == "__main__":
    main()
