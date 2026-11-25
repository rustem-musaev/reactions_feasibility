"""Step 2: fit the ISIDA fragmentor and store descriptors + labels in a shelve."""
import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from reaction_feasibility.descriptors import create_descriptors, fit_fragmentor  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="RDF file from step 1")
    p.add_argument("--shelve", default="descriptors.shelve", help="output descriptor store")
    p.add_argument("--fragmentor", default="fragmentor.pickle",
                   help="where to save the fitted fragmentor (or load it from, with --reuse)")
    p.add_argument("--reuse", action="store_true", help="load an already fitted fragmentor")
    p.add_argument("--skip-initial", action="store_true",
                   help="skip 'Initial' reactions instead of stopping with an error")
    a = p.parse_args()

    if a.reuse:
        with open(a.fragmentor, "rb") as f:
            fragmentor = pickle.load(f)
    else:
        fragmentor = fit_fragmentor(a.input, a.fragmentor)

    n = create_descriptors(a.input, fragmentor, a.shelve, skip_initial=a.skip_initial)
    print(f"Stored descriptors for {n} reactions in {a.shelve}")


if __name__ == "__main__":
    main()
