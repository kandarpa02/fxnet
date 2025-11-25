from .placeholder import FT_Tracer
from typing import NamedTuple, List
import numpy as np

def topo_sort(node):
    order = []
    visited = set()

    def dfs(n):
        if id(n) in visited:
            return
        visited.add(id(n))
        for p in n.parents:
            dfs(p)
        order.append(n)

    dfs(node)
    return order


Instruction = NamedTuple("Instruction", [
    ("func", callable),
    ("parent_ids", list[int]),
    ("out_id", int),
])


class CompiledFunction:
    def __init__(self, out_index, instrs, var_indices, num_nodes):
        self.out_index = out_index
        self.instrs = instrs
        self.var_indices = var_indices
        self.num_nodes = num_nodes

    def __call__(self, *args):
        buf = [None] * self.num_nodes

        # fill variable values
        for idx, val in zip(self.var_indices, args):
            buf[idx] = val

        # evaluate instructions
        for func, parent_ids, out_id in self.instrs:
            if len(parent_ids) == 0:
                continue
            elif len(parent_ids) == 1:
                buf[out_id] = func(buf[parent_ids[0]])
            else:
                buf[out_id] = func(*(buf[p] for p in parent_ids))

        return buf[self.out_index]


class FT_Function(NamedTuple):
    out: FT_Tracer
    variables: List[FT_Tracer]

    def compile(self):
        nodes = topo_sort(self.out)
        num_nodes = len(nodes)

        index = {id(n): i for i, n in enumerate(nodes)}
        var_indices = [index[id(v)] for v in self.variables]

        instrs = []
        for n in nodes:
            if n in self.variables:
                continue
            pid = [index[id(p)] for p in n.parents]
            instrs.append((n.func, pid, index[id(n)]))

        out_index = index[id(self.out)]
        return CompiledFunction(out_index, instrs, var_indices, num_nodes)
