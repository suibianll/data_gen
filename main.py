"""Entry point for the medical data generation pipeline.

Usage
-----
1. Place your medical dataset file(s) inside ``data/input/``.
2. Edit ``config.yaml`` to set your LLM API key, model, and generation parameters.
3. Run::

       python main.py [--input PATH] [--config PATH]

The pipeline will:
  1. Use an LLM to extract diseases, symptoms, and treatment methods from the
     input file.
  2. Generate synthetic Chinese medical admission records for each disease.
  3. Build Q&A pairs (with reference paragraphs) from each record and write
     the final dataset to ``data/output/qa_dataset.json``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from src.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Medical data generation pipeline"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help=(
            "Path to the input medical dataset file.  "
            "If not supplied, the first file found in config.paths.input_dir is used."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logging.error("Config file not found: %s", config_path)
        sys.exit(1)
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_input_file(args: argparse.Namespace, config: dict) -> Path:
    if args.input:
        p = Path(args.input)
        if not p.exists():
            logging.error("Input file not found: %s", p)
            sys.exit(1)
        return p

    input_dir = Path(config.get("paths", {}).get("input_dir", "data/input"))
    # Sort by name for deterministic, locale-independent ordering.
    candidates = sorted(input_dir.iterdir(), key=lambda p: p.name) if input_dir.exists() else []
    candidates = [f for f in candidates if f.is_file()]
    if not candidates:
        logging.error(
            "No input file specified and no files found in %s.\n"
            "Place your medical dataset file there or use --input PATH.",
            input_dir,
        )
        sys.exit(1)
    logging.info("No --input specified; using: %s", candidates[0])
    return candidates[0]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_file = resolve_input_file(args, config)

    pipeline = Pipeline(config)
    pipeline.run(input_file)


if __name__ == "__main__":
    main()
