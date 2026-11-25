"""Step 1: generate real + decoy reactions from a reaction database (RDF)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reaction_feasibility.generator import generate_dataset  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="input reactions, RDF file (atom-mapped)")
    p.add_argument("output", help="output RDF file with labelled reactions")
    p.add_argument("--limit", type=int, default=2, help="max applications of each rule (default 2)")
    p.add_argument("--batch", type=int, default=1000, help="write to disk every N reactions (default 1000)")
    p.add_argument("--total", type=int, default=0, help="only use the first N reactions (0 = all)")
    p.add_argument("--max-decoys", type=int, default=40, help="max generated reactions per input reaction")
    a = p.parse_args()

    conflicts = generate_dataset(a.input, a.output, limit=a.limit, batch=a.batch,
                                 total=a.total, max_decoys=a.max_decoys)
    print(f"Done. {len(conflicts)} label conflicts found.")


if __name__ == "__main__":
    main()
