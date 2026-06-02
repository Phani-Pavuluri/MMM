"""CLI for Bayes-H6 synthetic benchmark lane (research only)."""

from __future__ import annotations

import argparse
import json
import sys

from mmm.research.h6_synthetic.benchmark_harness import (
    build_h6_benchmark_artifact,
    build_h6_confounding_comparison,
)
from mmm.research.h6_synthetic.benchmark_matrix import (
    H6F_H5_WORLDS_DEFAULT,
    build_h6f_benchmark_matrix,
    build_h6f_control_confounding_summary,
)
from mmm.research.h6_synthetic.production_shapes import list_h6_world_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bayes-H6 synthetic validation (research only)")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="Ridge vs H5 benchmark on one world")
    bench.add_argument("world_id", choices=list(list_h6_world_ids()))
    bench.add_argument("--output", type=str, required=True)
    bench.add_argument("--no-h5", action="store_true", help="Ridge-only (faster)")
    bench.add_argument("--slow-mcmc", action="store_true")

    conf = sub.add_parser("confounding", help="H6d/H6f confounding stress comparison")
    conf.add_argument("--output", type=str, required=True)
    conf.add_argument("--with-h5", action="store_true")
    conf.add_argument("--h6f", action="store_true", help="Use H6f summary builder")

    matrix = sub.add_parser("matrix", help="H6f Ridge vs H5 benchmark matrix")
    matrix.add_argument("--output", type=str, required=True)
    matrix.add_argument("--with-h5", action="store_true", help="Run H5 on default pilot subset")
    matrix.add_argument("--slow-mcmc", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "benchmark":
        art = build_h6_benchmark_artifact(
            args.world_id,
            fast_mcmc=not args.slow_mcmc,
            run_h5=not args.no_h5,
        )
    elif args.command == "matrix":
        art = build_h6f_benchmark_matrix(
            h5_world_ids=H6F_H5_WORLDS_DEFAULT if args.with_h5 else (),
            fast_mcmc=not args.slow_mcmc,
        )
    else:
        if getattr(args, "h6f", False):
            art = build_h6f_control_confounding_summary(run_h5=args.with_h5)
        else:
            art = build_h6_confounding_comparison(run_h5=args.with_h5)

    path = args.output
    with open(path, "w", encoding="utf-8") as f:
        json.dump(art, f, indent=2, default=str)
    print(path, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
