import contextlib
import dataclasses
from typing import Callable, Sequence

COUNTER = -1

def new_ids():
    global COUNTER
    COUNTER+=1
    return COUNTER

@dataclasses.dataclass
class Tracer:
    id:int

@dataclasses.dataclass
class Node:
    func:Callable
    inputs:Sequence
    output:Tracer|Sequence[Tracer]


