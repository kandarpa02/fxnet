from ...tree_util import register_tree_node, flatten_pytree, unflatten_pytree, TreeDef, map
from typing import Dict, Union, OrderedDict, Any
from ...nn.parameters import Variable
import dataclasses
from ...src.ndarray.array_creation import zeros_like

ParamLike = Union[list[Variable]|tuple[Variable]|Dict]

class optimizer:
    def __init__(self, **kwargs) -> None:
        self._state = kwargs

    def init_state(self, params:ParamLike):
        state = {}
        for k, p in zip(self._state.keys(), params):
            state[k] = map(zeros_like, params)
        return state

    def state(self):
        return self._state
    
    def run_update(self, state, grads, params, **kwargs):
        return self.update_rule(state, grads, params, **kwargs)
    
    def update_rule(self, state, grads, params, **kwargs):
        raise NotImplementedError
    

def apply_updates(params:ParamLike, updates:ParamLike):
    param_list, pdef = flatten_pytree(params)
    update_list, udef = flatten_pytree(updates)
    new_param_list = []
    for p, u in zip(param_list, update_list):
        # new_param_list.append(p-u)
        new_param_list.append(p.assign(p-u))

    return unflatten_pytree(new_param_list, pdef)