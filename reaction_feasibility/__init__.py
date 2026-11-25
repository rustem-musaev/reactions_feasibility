"""Predicting the feasibility of organic reactions with a multilayer perceptron.

Pipeline:
    1. generator   -- build a labelled dataset of real ("Reconstructed") and
                      fake ("Decoy") reactions from a reaction database.
    2. descriptors -- turn each reaction (as a Condensed Graph of Reaction)
                      into ISIDA fragment counts.
    3. model       -- train a multilayer perceptron on those counts.
"""

__version__ = "1.0.0"
