"""Build a labelled dataset of real and decoy reactions.

For every reaction in the input RDF file:
  * extract its reaction rules (one per reaction centre),
  * apply those rules back to the original reactants,
  * label each generated reaction:
      - "Reconstructed" if it is the original reaction (rules recovered it),
      - "Decoy" if it is a new, presumably infeasible, reaction.
The original reaction is stored as "Initial" unless it gets reconstructed.
"""
import os

from CGRtools.containers import ReactionContainer
from CGRtools.files import RDFRead, RDFWrite

from .utils import containers_split, generate_reactions, not_radical, remove_reagents

INITIAL = {"type": "Initial"}
RULE = {"type": "Rule"}
RECONSTRUCTED = {"type": "Reconstructed"}
DECOY = {"type": "Decoy"}


def get_rules_and_decoys(reaction, limit=2, max_decoys=40):
    """Return {signature: {"structure": reaction, "type": label}} for one reaction."""
    d = {}
    reaction.meta.update(INITIAL)
    d[str(reaction)] = {"structure": reaction, "type": reaction.meta["type"]}
    initial_reactants = reaction.reactants
    rules = {}

    for partial_reaction in reaction.enumerate_centers():
        reaction_center = set(partial_reaction.extended_centers_list[0])
        bare_reaction_center = set(partial_reaction.compose().center_atoms)
        reactants = []
        products = []
        for side, groups in ((partial_reaction.reactants, reactants),
                             (partial_reaction.products, products)):
            for mol in side:
                group_atoms = reaction_center.intersection(mol)
                if group_atoms:
                    group = mol.substructure(group_atoms, as_query=True)
                    # Atoms outside the bare centre: drop neighbour-count constraint.
                    for i in group_atoms.difference(bare_reaction_center):
                        group._neighbors[i] = ()
                    groups.append(group)
        rule = ReactionContainer(reactants=reactants, products=products, meta=reaction.meta)
        rule.meta.update(RULE)
        rules[str(rule)] = {"structure": rule, "type": rule.meta["type"]}

    if rules and len(rules) < 4:
        unique_rules = list({x["structure"] for x in rules.values()})
        try:
            for c, new_reaction in enumerate(generate_reactions(initial_reactants, unique_rules, limit)):
                if new_reaction is None:
                    continue
                new_reaction.meta.update(reaction.meta)
                if c == max_decoys:
                    return d
                signature = str(new_reaction)
                if signature not in d:
                    new_reaction.meta.update(DECOY)
                elif d[signature]["type"] == "Initial":
                    new_reaction.meta.update(RECONSTRUCTED)
                else:
                    continue
                d[signature] = {"structure": new_reaction, "type": new_reaction.meta["type"]}
        except KeyError:
            reaction.meta.update({"Error": "KeyError"})
            print("ERROR: KeyError while generating decoys for", reaction.meta)
            return d
    return d


def _write(output, reactions):
    with open(output, "a") as w, RDFWrite(w) as rdf:
        for item in reactions:
            item.clean2d()
            rdf.write(item)


def generate_dataset(filename, output, limit=2, batch=1000, total=0, max_decoys=40):
    """Process `filename` (RDF) and write labelled reactions to `output` (RDF).

    limit       -- max times each rule may be applied along one path
    batch       -- write to disk every `batch` input reactions
    total       -- only process the first `total` reactions (0 = all)
    max_decoys  -- max generated reactions per input reaction

    Returns a list of signature conflicts (same reaction, different labels).
    """
    try:
        os.remove(output)
    except FileNotFoundError:
        pass

    conflict = []
    signatures = {}
    temp = []
    with RDFRead(filename, indexable=True) as f:
        data = f if total == 0 else f[:total]
        for n, reaction in enumerate(data, start=1):
            reaction = remove_reagents(reaction)
            if reaction is not None and len(reaction.reactants) < 3 and str(reaction) not in signatures:
                reaction = containers_split(reaction)
                cgr = ~reaction
                if not_radical(cgr):
                    d = get_rules_and_decoys(reaction, limit, max_decoys)
                    for duplicate in set(signatures).intersection(d):
                        if d[duplicate]["type"] != signatures[duplicate]["type"]:
                            conflict.append((duplicate, (d[duplicate]["type"], signatures[duplicate]["type"])))
                        del d[duplicate]
                    for signature, entry in d.items():
                        temp.append(entry["structure"])
                        signatures[signature] = {"type": entry["type"]}

            if n % batch == 0:
                print(n, "reactions processed")
                _write(output, temp)
                temp = []

    _write(output, temp)
    return conflict
