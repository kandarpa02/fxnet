from ..base import MakeOP

class Op:
    def __init__(self) -> None:
        self.parents = None
    
    def build(self):
        @MakeOP
        def fun(*args):
            out = self.forward(*args)
            parents = args if self.parents is None else self.parents
            if self.parents is None:
                self.parents = list(parents)
            grad_fn = lambda grad:self.backward(grad)
            return out, parents, grad_fn
        return fun

    def add_parents(self, *args):
        self.parents = list(args)

    def get_parents(self):
        return self.parents[0] if len(self.parents)==1 else tuple(self.parents)

    def forward(self, *args):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    def apply(self, *args):
        return self.build()(*args)