"""Reaction clean-up and reaction-rule application helpers (CGRtools)."""
from itertools import combinations

from CGRtools.containers import ReactionContainer
from CGRtools.exceptions import MappingError
from CGRtools.reactor import Reactor


def remove_reagents(reaction):
    """Remove molecules that do not change during the reaction (reagents, solvents).

    Returns a new ReactionContainer, or None if the reaction cannot be turned
    into a CGR or nothing is left on one side after clean-up.
    """
    try:
        cgr = ~reaction
    except ValueError:
        return None

    # Atoms of CGR components with no charge change and no bond change
    # belong to spectator molecules.
    to_remove = []
    for component in cgr.split():
        if any(atom.charge != atom.p_charge for _, atom in component.atoms()) or \
                any(bond.order != bond.p_order for _, _, bond in component.bonds()):
            continue
        to_remove.extend(n for n, _ in component.atoms())

    for molecule in reaction.molecules():
        for atom in to_remove:
            try:
                molecule.delete_atom(atom)
            except KeyError:
                continue
    reaction.flush_cache()

    reactants = [x for x in reaction.reactants if x]
    products = [x for x in reaction.products if x]
    if reactants and products:
        return ReactionContainer(reactants=reactants, products=products, meta=reaction.meta)
    return None


def not_radical(cgr):
    """True unless an atom in a reaction with a reaction centre is a radical."""
    if cgr and cgr.center_atoms:
        if any(atom.is_radical or atom.p_is_radical for _, atom in cgr.atoms()):
            return False
    return True


def containers_split(reaction):
    """Split multi-component containers so every molecule is its own container."""
    new_reactants = [x for container in reaction.reactants for x in container.split()]
    new_products = [x for container in reaction.products for x in container.split()]
    return ReactionContainer(reactants=new_reactants, products=new_products, meta=reaction.meta)


def generate_reactions(reactants, rules, limit=5):
    """Yield single-product reactions obtained by applying `rules` to `reactants`.

    Only reactions with a real reaction centre are yielded.
    """
    for result in apply_rules(reactants, rules, limit):
        for product in result.products:
            new_reaction = ReactionContainer(reactants=reactants, products=[product])
            try:
                if (~new_reaction).center_atoms:
                    yield new_reaction
            except MappingError:
                continue


def apply_rules(reactants, rules, limit=2):
    """Breadth-first application of reaction rules.

    Each rule may be applied at most `limit` times along one path. Products of
    one step are fed back in, alone and combined with the original reactants.
    """
    max_fusion = [len(rule.reactants) - 1 for rule in rules]
    reactors = [Reactor(rule, delete_atoms=True) for rule in rules]
    seen = set()
    queue = [(reactor(reactants, False), [0] * len(rules)) for reactor in reactors]
    while queue:
        results, counts = queue.pop(0)
        for result in results:
            if result in seen:
                continue
            seen.add(result)
            yield result
            for num, (used, reactor, m_f) in enumerate(zip(counts, reactors, max_fusion)):
                if used != limit:
                    new_counts = counts.copy()
                    new_counts[num] += 1
                    queue.append((reactor(result.products, False), new_counts))
                    for extra in combinations(reactants, m_f):
                        queue.append((reactor(result.products + extra, False), new_counts))
