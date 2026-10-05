"""Package a model release, or install a specific checksum-verified bundle."""
import argparse
import json
from serving.bundle import package, install, download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--artifact", required=True)
    pack.add_argument("--out", required=True)
    fetch = commands.add_parser("install")
    source = fetch.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive")
    source.add_argument("--url")
    fetch.add_argument("--sha256", required=True)
    fetch.add_argument("--destination", required=True)
    args = parser.parse_args()
    if args.command == "pack":
        print(json.dumps(package(args.artifact, args.out), indent=2))
    else:
        print(download(args.url, args.destination, args.sha256) if args.url else
              install(args.archive, args.destination, args.sha256))


if __name__ == "__main__":
    main()
