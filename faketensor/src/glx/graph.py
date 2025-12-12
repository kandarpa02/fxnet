from .base import Proxy
from .graph_op import OpRegister

class Graph(Proxy):
    def __init__(self, name:str, inputs:list, out, **kwargs) -> None:
        super().__init__()
        self.name = name
        self.params = kwargs
        self.func = lambda *args:OpRegister.get(name, lambda:None)(*args, **kwargs)
        self.inputs = inputs
        self.out = out
    
    def repr(self):
        return f'Graph({self.name}, {self.inputs}, {self.out})'
    
class CompositeGraph:
    def __init__(self):
        self.nodes = []   # list of Graph objects
    
    def add(self, node: Graph):
        self.nodes.append(node)

def compile_graph(cgraph: CompositeGraph):
    """
    Compile the entire composite graph (DAG) into a static executable function.
    """

    # Extract ops in order
    op_list = []
    for node in cgraph.nodes:
        op_list.append((
            node.func,                         # lambda for Add, Mul, etc.
            [i.name for i in node.inputs],     # input variable names
            node.out.name                      # output variable name
        ))

    def compiled_fn(feed_dict: dict[str, np.ndarray]):
        env = dict(feed_dict)   # execution environment

        for op, in_names, out_name in op_list:
            # Resolve actual inputs
            args = [env[n] for n in in_names]
            # Execute
            env[out_name] = op(*args)

        return env  # contains all intermediates + final outputs

    return compiled_fn