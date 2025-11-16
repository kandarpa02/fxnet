# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3 


cdef class Node:
    cdef object _out
    cdef tuple _parents
    cdef object _backward

    def __init__(self, object out, tuple parents, object backward):
        self._out = out
        self._parents = parents
        self._backward = backward

    @property
    def out(self):
        return self._out

    @property
    def parents(self):
        return self._parents

    @property
    def backward(self):
        return self._backward

def create_node(out, parents, backward):
    return Node(out, parents, backward)

