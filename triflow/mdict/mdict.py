#!/usr/bin/env python
# coding=utf8

import collections
import sympy as sp
from queue import Queue
import logging
from toolz import memoize
from copy import deepcopy

logging.getLogger(__name__).addHandler(logging.NullHandler())
logging = logging.getLogger(__name__)

MagicRelation = collections.namedtuple('MagicRelation', ('atoms', 'rel'))


class magical_dict(collections.MutableMapping):

    def __init__(self, *args, **kwargs):
        self.magic_relations = []
        self.store = dict()
        self.deduced = dict()
        self.safe = False

    def manage_conflict(self, new_key, queue):
        conflict_keys = [key
                         for key, value
                         in self.deduced.items()
                         if new_key in value['from']]
        if self.safe and len(conflict_keys) > 0:
            raise KeyError('Conflict detected between %s and %s' %
                           (new_key, ' - '.join(conflict_keys)))
        for key in conflict_keys:
            logging.info('key %s in conflict, delete and recompute' % key)
            del self.deduced[key]
            queue.put(key)

    def get_deduced(self, key, queue):
        concerned_relations = [mrel for mrel in self.magic_relations
                               if key in [str(atom)
                                          for atom in mrel.atoms]]
        for concerned_relation in concerned_relations:
            available_keys = set(map(str,
                                     concerned_relation.atoms)
                                 ).difference(self.chain_dict.keys())
            if len(available_keys) == 1:
                deduced_key = available_keys.pop()
                linked_keys = set(map(str,
                                      concerned_relation
                                      .atoms))
                linked_keys.remove(deduced_key)
                solved = sp.solve(
                    concerned_relation.rel
                    .subs(self.chain_dict.items())
                )
                value = float(solved[0])
                queue.put(deduced_key)
                yield deduced_key, linked_keys, value

    def __setitem__(self, key, value):
        queue = Queue()
        queue.put(key)
        self.store[key] = value
        self.manage_conflict(key, queue)
        while not queue.empty():
            for deduced_key, linked_keys, value in\
                    self.get_deduced(queue.get(), queue):
                logging.info('key %s deduced from %s' % (deduced_key,
                                                         linked_keys))
                self.deduced[deduced_key] = {}
                self.deduced[deduced_key]['from'] = linked_keys
                self.deduced[deduced_key]['value'] = value

    def __missing__(self, key):
        deduced_value = self.deduced[key]
        return deduced_value

    def add_relation(self, relation):
        relation = sp.S(relation)
        magic_relation = MagicRelation(rel=relation,
                                       atoms=relation.atoms(sp.Symbol))
        self.magic_relations.append(magic_relation)

    def __iter__(self):
        return iter(self.chain_dict)

    def __getitem__(self, key):
        return self.chain_dict[key]

    def __delitem__(self, key):
        del self.store[key]

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return self.chain_dict.__repr__()

    def __copy__(self):
        copy = magical_dict()
        copy.magic_relations = deepcopy(self.magic_relations)
        copy.store = deepcopy(self.store)
        copy.deduced = deepcopy(self.deduced)
        return copy

    def copy(self):
        return self.__copy__()

    @property
    def chain_dict(self):
        return collections.ChainMap(self.store,
                                    {key: value['value']
                                     for key, value
                                     in self.deduced.items()})
