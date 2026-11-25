# Reaction feasibility with a multilayer perceptron

This is the code from my bachelor's thesis (2021). It trains a small neural
network, a multilayer perceptron, to tell real organic reactions from
made-up ("decoy") ones. Each reaction is treated as a graph: a
**Condensed Graph of Reaction (CGR)**, which merges reactants and products
into one graph that marks the bonds that form and break.

> Thesis: An instrument for predicting the feasibility of organic reactions for retrosynthesis, Kazan Federal University, 2020. Author: Rustem Musaev.

## How it works

The pipeline has three steps.

**1. Build the dataset** (`reaction_feasibility/generator.py`)
- Read atom-mapped reactions from an RDF file.
- Clean them up: remove reagents and solvents (molecules that don't change),
  skip reactions with 3 or more reactants, duplicates, and radical reactions.
- For each reaction, extract its **reaction rules**: the reaction centre plus
  its nearest neighbours, one rule per centre. Reactions with 4 or more
  centres are skipped.
- Apply those rules back to the original reactants (each rule at most twice
  along one path, up to 40 results per reaction). Each result gets a label:
  - **Reconstructed**: the rules gave back the original reaction (a real reaction).
  - **Decoy**: a new reaction the rules allow but that isn't recorded (treated as not feasible).
- Save everything to a new RDF file. The original reaction is saved as
  **Initial** if the rules could not reconstruct it.

**2. Compute descriptors** (`reaction_feasibility/descriptors.py`)
- Turn each CGR into a vector of **ISIDA fragment counts** (CIMtools
  `Fragmentor`, fragment type 10, 1–5 atoms long).
- Store each vector with its label (0 = Reconstructed, 1 = Decoy) in a
  compressed on-disk store.

**3. Train the network** (`reaction_feasibility/model.py`, `scripts/train.py`)
- Architecture: input → batch normalisation → 1000 ReLU neurons →
  batch normalisation → 1 sigmoid output.
- Loss: binary cross-entropy, Adam optimiser, batch size 200.
- Data split: 10 % test (seed 42); the rest split 80 / 20 into train and
  validation (seed 41).
- Tracked metrics: F1 score, balanced accuracy and ROC AUC. Training stops
  early when validation balanced accuracy stops improving.

## Layout

```
reaction_feasibility/   the code as an importable package
  utils.py              reaction clean-up, rule application
  generator.py          step 1: real + decoy reactions
  descriptors.py        step 2: ISIDA fragment descriptors
  compress.py           compressed on-disk store (stand-in, see below)
  model.py              step 3: network, metrics, batch generator
scripts/                command-line entry points for each step
original/               the thesis code exactly as it was run (for reference)
```

## Install

You need Python 3.7 or 3.8. CIMtools also needs the ISIDA Fragmentor
program installed separately; see the CIMtools documentation.

```
pip install -r requirements.txt
```

## Run

Each step is one command:

```
python scripts/generate_dataset.py reactions.rdf dataset.rdf
python scripts/compute_descriptors.py dataset.rdf --skip-initial --shelve descriptors.shelve
python scripts/train.py --shelve descriptors.shelve
```

Add `--help` to any script to see all options. Training logs can be viewed
with `tensorboard --logdir logs`.

## How this differs from the thesis code

`original/` holds the code as it was printed in 2020, typed back in from the
PDFs (only whitespace is tidied). The package in `reaction_feasibility/` is
the same method, cleaned up so it runs as a whole. The changes:

- **Missing imports added**, and fixed file names are now command-line
  options.
- **Best-model checkpoint fixed.** The original saved the checkpoint with
  `mode='min'` on balanced accuracy, so it kept the *worst* epoch. It now
  uses `mode='max'`.
- **Stale results no longer repeated.** In the original `main()`, a radical
  reaction re-added the previous reaction's results to the output. Results
  are now only added for reactions that pass the radical check.
- **Error message fixed.** `raise(print(...))` became a proper `ValueError`.
  The new `--skip-initial` option drops "Initial" reactions instead. In the
  thesis these were removed before step 2 (the files were named
  `..._wo_initials`).
- **Test-set evaluation fixed.** `create_model()` in the original called
  `model.predict` as if it returned metrics. `train.py` now uses
  `model.evaluate` on the test set.
- **Duplicate data generators merged.** `generator`, `new_generator` and
  `new_generator_w_graph` became one `batch_generator`.
- **Old Keras argument renamed.** `epsilon` became `min_delta`.
- **`compress.py` is a stand-in.** The original `CompressShelve` came from a
  separate module that wasn't archived. This version has the same interface
  but may not read shelve files written by the original.
