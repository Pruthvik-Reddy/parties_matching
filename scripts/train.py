from __future__ import annotations

import argparse
import os

from party_matching.domain import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the feature matcher and optional selective cross-encoder.")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--cross-encoder", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    native_threads = str(config.get("execution", {}).get("native_threads", 10))
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, native_threads)
    from party_matching.matching import train_models
    if args.cross_encoder:
        config.setdefault("training", {})["train_cross_encoder"] = True
    report = train_models(config)
    print(f"Trained on {report['training_pairs']:,} pairs; system threshold={report['system_threshold']:.4f}; cross-encoder={report['cross_encoder']}")


if __name__ == "__main__":
    main()
