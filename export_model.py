"""Package a full-trained winner for portable CPU inference."""
import argparse
from serving.artifact import export_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--example", help="Optional prediction-request JSON bundled for the demo")
    args = parser.parse_args()
    manifest = export_artifact(args.results, args.out, args.example)
    print(f"Exported {manifest['model_id']} to {args.out}")


if __name__ == "__main__":
    main()
