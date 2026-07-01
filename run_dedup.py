#!/usr/bin/env python3
"""CLI entrypoint for the OpenSearch vector dedup test demo.

Examples
--------
# Validate the whole pipeline locally (no cluster needed), 100k vectors:
python run_dedup.py --dry-run --total 100000

# Run against a real OpenSearch domain, 1亿 (100M) vectors:
OS_HOST=my-domain.us-west-2.es.amazonaws.com \\
OS_USERNAME=admin OS_PASSWORD=*** \\
python run_dedup.py --total 100000000 --batch-size 20000

# Managed Amazon OpenSearch Service with SigV4:
OS_HOST=my-domain.us-west-2.es.amazonaws.com OS_USE_AWS_AUTH=true \\
OS_AWS_REGION=us-west-2 \\
python run_dedup.py --total 100000000
"""
from __future__ import annotations

import argparse
import json
import sys

from src.config import Config
from src.dedup_runner import run


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenSearch sequential-batch vector dedup demo")
    p.add_argument("--backend", choices=["opensearch", "pgvector"],
                   help="vector engine: opensearch (default) or pgvector (RDS PostgreSQL)")
    p.add_argument("--total", type=int, help="total vectors to process (default 100M)")
    p.add_argument("--batch-size", type=int, help="vectors per batch (default 20000)")
    p.add_argument("--dup-ratio", type=float, help="fraction of near-duplicates (default 0.30)")
    p.add_argument("--dim", type=int, help="vector dimension (default 768)")
    p.add_argument("--index", type=str, help="index name (default video_vectors)")
    p.add_argument("--dry-run", action="store_true",
                   help="skip OpenSearch; use ground-truth oracle to validate the pipeline")
    p.add_argument("--seed", type=int, help="RNG seed (default 42)")
    p.add_argument("--progress-every", type=int, default=10, help="print progress every N batches")
    p.add_argument("--report-every-batch", action="store_true",
                   help="print per-batch phase timing (local/search/write/refresh) to spot "
                        "whether search or insert cost grows as the index fills up")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config()

    if args.backend is not None:
        cfg.backend = args.backend
    if args.total is not None:
        cfg.total_vectors = args.total
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.dup_ratio is not None:
        cfg.duplicate_ratio = args.dup_ratio
    if args.dim is not None:
        cfg.dimension = args.dim
    if args.index is not None:
        cfg.index_name = args.index
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.dry_run = args.dry_run

    print("=" * 68)
    print("OpenSearch Vector Dedup — Test Demo")
    print("=" * 68)
    print(f"  mode          : {'DRY-RUN (no cluster)' if cfg.dry_run else 'LIVE'}")
    print(f"  backend       : {cfg.backend}")
    print(f"  total vectors : {cfg.total_vectors:,}")
    print(f"  batch size    : {cfg.batch_size:,}")
    print(f"  dup ratio     : {cfg.duplicate_ratio}")
    print(f"  dimension     : {cfg.dimension}")
    print(f"  cosine thresh : {cfg.cosine_threshold} (min_score {cfg.min_score})")
    if not cfg.dry_run:
        if cfg.backend.lower() in ("opensearch", "os"):
            print(f"  host          : {cfg.host}:{cfg.port}")
            print(f"  index         : {cfg.index_name}")
        else:
            print(f"  host          : {cfg.pg_host}:{cfg.pg_port}/{cfg.pg_db}")
            print(f"  table         : {cfg.pg_table}")
    print("-" * 68)

    stats = run(cfg, progress_every=args.progress_every,
                report_every_batch=args.report_every_batch)

    print("=" * 68)
    print("RESULT")
    print("=" * 68)
    print(json.dumps(stats.summary(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
