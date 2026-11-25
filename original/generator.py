# Transcribed from GENERATOR_CODE.pdf (printed 27/11/2020).
# Kept as it was used; see reaction_feasibility/generator.py for the maintained version.
from CGRtools.files import RDFRead
from CGRtools.containers import ReactionContainer
from new_generator.utils import generate_reactions, remove_reagents, containers_split, not_radical
from CGRtools.files import RDFWrite
import os

INITIAL = {"type": "Initial"}
RULE = {"type": "Rule"}
RECONSTRUCTED = {"type": "Reconstructed"}
DECOY = {"type": "Decoy"}


def get_rules_and_decoys(reaction, limit=2, max_decoys=40):
    d = {}
    reaction.meta.update(INITIAL)
    d.update({str(reaction): {"structure": reaction, "type": reaction.meta["type"]}})
    initial_reactants = reaction.reactants
    rules = {}
    #print(reaction.meta)
    #try:
    for n, partial_reaction in enumerate(reaction.enumerate_centers()):
        reaction_center = set(partial_reaction.extended_centers_list[0])
        bare_reaction_center = set(partial_reaction.compose().center_atoms)
        reactants = []
        products = []
        for mol in partial_reaction.reactants:
            group_atoms = reaction_center.intersection(mol)
            if group_atoms:
                group = mol.substructure(group_atoms, as_query=True)
                for i in group_atoms.difference(bare_reaction_center):
                    group._neighbors[i] = ()
                reactants.append(group)
        for mol in partial_reaction.products:
            group_atoms = reaction_center.intersection(mol)
            if group_atoms:
                group = mol.substructure(group_atoms, as_query=True)
                for i in group_atoms.difference(bare_reaction_center):
                    group._neighbors[i] = ()
                products.append(group)
        rule = ReactionContainer(reactants=reactants, products=products,
                                 meta=reaction.meta)
        rule.meta.update(RULE)
        rules.update({str(rule): {"structure": rule, "type": rule.meta["type"]}})
    #except ValueError:
        #return d
    if rules and len(rules) < 4:
        try:
            for c, new_reaction in enumerate(generate_reactions(initial_reactants, list(set([x["structure"] for x in rules.values()])), limit)):

                if new_reaction is not None:
                    new_reaction.meta.update(reaction.meta)
                    if c == max_decoys:
                        return d
                    if str(new_reaction) not in d:
                        new_reaction.meta.update(DECOY)
                    elif str(new_reaction) in d and d[str(new_reaction)]["type"] == "Initial":
                        new_reaction.meta.update(RECONSTRUCTED)
                    else:
                        continue
                    #print(new_reaction.meta)
                    d.update({str(new_reaction): {"structure": new_reaction, "type": new_reaction.meta["type"]}})
        except KeyError:
            reaction.meta.update({"Error": "KeyError"})
            print("ERROR !!!!!!!!!! ", reaction.meta)
            return d
    return d


def main(limit, filename, output, batch, total=0):
    try:
        os.remove(output)
    except FileNotFoundError:
        pass
    conflict = []
    signatures = {}
    temp = []
    with RDFRead(filename, indexable=True) as f:
        if total == 0:
            data = f
        else:
            data = f[:total]
        for n, reaction in enumerate(data, start=1):
            reaction = remove_reagents(reaction)
            if reaction is not None and len(reaction.reactants) < 3 and str(reaction) not in signatures:
                reaction = containers_split(reaction)
                cgr = ~reaction
                if not_radical(cgr):
                    d = get_rules_and_decoys(reaction, limit)
                    intersection = set(signatures).intersection(set(d))
                    for duplicate in intersection:
                        if d[duplicate]["type"] != signatures[duplicate]["type"]:
                            conflict.append((duplicate, (d[duplicate]["type"], signatures[duplicate]["type"])))
                        del d[duplicate]
                for new_reaction in d:
                    temp.append(d[new_reaction]["structure"])
                    signatures.update({new_reaction: {"type": d[new_reaction]["type"]}})

            if n % batch == 0:
                print(n)
                with open(output, "a") as w, RDFWrite(w) as rdf:
                    for item in temp:
                        item.clean2d()
                        #print(item.meta)
                        rdf.write(item)
                temp = []
        else:
            with open(output, "a") as w, RDFWrite(w) as rdf:
                for item in temp:
                    item.clean2d()
                    #print(item.meta)
                    rdf.write(item)
    return conflict
