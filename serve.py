"""Serve a local experiment dashboard and portable model predictions."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=os.environ.get("NAS_ARTIFACT"))
    parser.add_argument("--report", default=os.environ.get("NAS_REPORT"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=os.environ.get("NAS_DEVICE", "cuda"), help="Inference device (default: cuda)")
    parser.add_argument("--threads", type=int, default=1, help="CPU inference threads")
    args = parser.parse_args()
    import uvicorn
    import torch
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    from serving.api import create_app
    uvicorn.run(create_app(args.artifact, args.report, device=args.device), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
