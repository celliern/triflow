import sympy as sp
from logging import info


import sympy as sp
from logging import info


def rebuild_magical_dict(this_dict, magic_relations):
    this_dict = magical_dict(this_dict)
    for relation in magic_relations:
        this_dict.add_relation(relation)
    return this_dict

class magical_dict(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.magic_relations = []

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        sols = None
        while sols != []:
            appropriate_magic_relation = [magic_relation.subs(self.items())
                                          for magic_relation
                                          in self.magic_relations]
            appropriate_magic_relation = [appr_rel for appr_rel
                                          in appropriate_magic_relation
                                          if len(appr_rel.atoms(sp.Symbol)
                                                 .difference(
                                              set(
                                                  map(sp.Symbol,
                                                      self.keys()
                                                      )
                                              )
                                          )
                                          ) == 1]
            sols = sp.solve(appropriate_magic_relation, dict=True)
            for sol in sols:
                sol = {str(solkey): float(solvalue)
                       for solkey, solvalue in sol.items()}

                [info("%s deduced from relations, value : %.2E" % (
                    solkey, solvalue)) for solkey, solvalue in sol.items()]
                self.update(sol)

    def __missing__(self, key):
        appropriate_magic_relation = [magic_relation.subs(self.items())
                                      for magic_relation
                                      in self.magic_relations
                                      if sp.Symbol(key)
                                      in magic_relation.atoms(sp.Symbol)]
        appropriate_magic_relation = [appr_rel for appr_rel
                                      in appropriate_magic_relation
                                      if appr_rel.atoms(sp.Symbol).difference(
                                          set(
                                              map(
                                                  sp.Symbol,
                                                  self.keys()
                                              )
                                          )
                                      ) == {sp.Symbol(key)}]
        sol = sp.solve(appropriate_magic_relation, sp.Symbol(key), dict=True)
        if sol != []:
            try:
                self[key] = float(sol[0][sp.Symbol(key)])
                return float(sol[0][sp.Symbol(key)])
            except TypeError:
                parameters = sol[0][sp.Symbol(key)].atoms(
                    sp.Symbol).difference(set(map(sp.Symbol, self.keys())))
                raise KeyError(("%s not in dict," % key +
                                "yet solvable with provided relation," +
                                "but a necessary value is missing. " +
                                "should be " + ', '
                                .join(map(str,
                                          list(parameters))) + '.'))
        else:
            raise KeyError(
                '%s not in dict and not solvable with provided relations' % key
            )

    def add_relation(self, relation):
        self.magic_relations.append(sp.S(relation))

    def __reduce__(self):
        return (rebuild_magical_dict, (dict(self), self.magic_relations))
