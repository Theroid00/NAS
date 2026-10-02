"""Create a portable JSON report for the results dashboard."""
import argparse
from reporting.comparison import save_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--out", default="docs/industry-results.json")
    args = parser.parse_args()
    report = save_report(args.comparison, args.out)
    print(f"Report saved: {args.out}; {len(report['runs'])} searches recorded")


if __name__ == "__main__":
    main()
