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


class FT_Function(NamedTuple):
    out: FT_Tracer
    variables: List[FT_Tracer]

    def feed(self, *args):
        assert len(args) == len(self.variables), "Wrong number of arguments"
        env = {}

        for tr, val in zip(self.variables, args):
            env[id(tr)] = np.asarray(val)

        # 2) Topologically sort the entire graph
        nodes = topo_sort(self.out)

        # 3) Evaluate each node
        for n in nodes:
            if id(n) in env:
                continue  # Variable already has runtime value

            if n.is_leaf():
                raise ValueError("Leaf node has no runtime value")

            # evaluate function
            parent_values = [env[id(p)] for p in n.parents]
            result = n.func(*parent_values)
            env[id(n)] = result

        # 4) Return final output
        return env[id(self.out)]
