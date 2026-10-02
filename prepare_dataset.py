"""Download the selected research dataset once, before starting a search."""
import argparse
import json
from data.specs import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), default="covertype")
    args = parser.parse_args()
    from data.tabular import prepare_dataset
    print(json.dumps(prepare_dataset(args.dataset), indent=2))


if __name__ == "__main__":
    main()
