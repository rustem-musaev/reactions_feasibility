"""ISIDA fragment descriptors of reactions (as Condensed Graphs of Reaction)."""
import pickle

from CGRtools.files import RDFRead
from CIMtools.preprocessing import Fragmentor

from .compress import CompressShelve

# Class labels: 0 = real (reconstructed) reaction, 1 = decoy.
LABELS = {"Reconstructed": 0, "Decoy": 1}


def fit_fragmentor(filename, output_pickle="fragmentor.pickle", chunk=1000):
    """Fit an ISIDA Fragmentor on all reactions in `filename` and pickle it."""
    fragmentor = Fragmentor(fragment_type=10, max_length=5, min_length=1, cgr_dynbonds=0)
    with RDFRead(filename, indexable=True) as f:
        n_chunks = len(f) // chunk + bool(len(f) % chunk)
        for i in range(n_chunks):
            fragmentor.fit([~x for x in f[i * chunk:(i + 1) * chunk]])
        fragmentor.finalize()

    with open(output_pickle, "wb") as p:
        pickle.dump(fragmentor, p)
    return fragmentor


def create_descriptors(filename, fragmentor, shelve_path="descriptors.shelve",
                       chunk=100, skip_initial=False):
    """Compute descriptors and store {"<index>": [1-row DataFrame, label]} in a shelve.

    The input should contain only "Reconstructed" and "Decoy" reactions.
    With skip_initial=True, "Initial" reactions are skipped instead of raising.
    """
    index = 0
    with RDFRead(filename, indexable=True) as f, CompressShelve(shelve_path) as shelve:
        n_chunks = len(f) // chunk + bool(len(f) % chunk)
        for k in range(n_chunks):
            cgrs_list = []
            class_values = []
            for reaction in f[k * chunk:(k + 1) * chunk]:
                kind = reaction.meta["type"]
                if "Reconstructed" in kind:
                    class_values.append(LABELS["Reconstructed"])
                elif "Decoy" in kind:
                    class_values.append(LABELS["Decoy"])
                elif skip_initial and "Initial" in kind:
                    continue
                else:
                    raise ValueError(f"Unexpected reaction type: {reaction.meta}")
                cgrs_list.append(~reaction)
            if not cgrs_list:
                continue
            descriptor = fragmentor.transform(cgrs_list)
            if len(descriptor) == len(cgrs_list) == len(class_values):
                for n, class_value in enumerate(class_values):
                    shelve[str(index)] = [descriptor.iloc[[n]], class_value]
                    index += 1
    return index
