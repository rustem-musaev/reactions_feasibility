# Transcribed from UTILS_CODE.pdf (printed 27/11/2020).
# Imported in the thesis code as `new_generator.utils`.
# Kept as it was used; see reaction_feasibility/utils.py for the maintained version.
from itertools import combinations
from CGRtools.reactor import Reactor
from CGRtools.containers import ReactionContainer
from CGRtools.exceptions import MappingError


def remove_reagents(reaction):
    try:
        cgr = ~reaction
    except ValueError:
        return None
#        return reaction
    to_remove = []
    for n, i in enumerate(cgr.split()):
        if any([z[1].charge != z[1].p_charge for z in i.atoms()]) or \
         any([z[2].order != z[2].p_order for z in i.bonds()]):
            continue
        to_remove.extend([x for x, _ in i.atoms()])
    for i in reaction.molecules():
        for z in to_remove:
            try:
                i.delete_atom(z)
#                print(z)
            except KeyError:
                #print(z)
                continue
    reaction.flush_cache()
    reactants = [x for x in reaction.reactants if x]
    products = [x for x in reaction.products if x]
    if len(reactants) > 0 and len(products) > 0:
        reaction = ReactionContainer(reactants=reactants,
                                     products=products, meta=reaction.meta)
        return reaction
    else:
        return None


def not_radical(cgr):
    if cgr and cgr.center_atoms:
        if any(x.is_radical or x.p_is_radical for _, x in cgr.atoms()):
            return False
    return True

def containers_split(reaction):
    new_reactants = [x for container in reaction.reactants for x in container.split()]
    new_products = [x for container in reaction.products for x in container.split()]
    return ReactionContainer(reactants=new_reactants, products=new_products, meta=reaction.meta)


def generate_reactions(reactants, rules, limit=5):
    #try:
    for r in apply_rules(reactants, rules, limit):
        for i in r.products:
            new_reaction = ReactionContainer(reactants=reactants, products=[i])
            try:
                if (~new_reaction).center_atoms:
                    yield new_reaction
                else:
                    continue
            except MappingError:
                continue
        #yield ReactionContainer(reactants=reactants, products=r.products)
    #except KeyError or ValueError:
    #     print(re)
      #   return None

def apply_rules(reactants, rules, limit=2):
    max_fusion = [len(rule.reactants) - 1 for rule in rules]
    reactors = [Reactor(x, delete_atoms=True) for x in rules]
    seen = set()
    queue = [(r(reactants, False), [0]*len(rules)) for r in reactors]
    while queue:
        r, n = queue.pop(0)
        for i in r:
            if i in seen:
                continue
            seen.add(i)
            yield i
            for num, (l, reactor, m_f) in enumerate(zip(n, reactors, max_fusion)):
                if l != limit:
                    n_new = n.copy()
                    n_new[num] += 1
                    queue.append((reactor(i.products, False), n_new))
                    for x in combinations(reactants, m_f):
                        queue.append((reactor(i.products+x, False), n_new))
