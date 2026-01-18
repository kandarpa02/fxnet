from .base import optimizer, ParamLike
from ...src.tree_util import flatten_pytree, unflatten_pytree

class sgd(optimizer):
    def __init__(self, 
        lr: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float = 0.0
                ) -> None:
        
        _state = {"velocity":None}

        super().__init__(**_state)
        self.lr = lr
        self.momentum = momentum
        self.nestrov = nesterov
        self.weight_decay = weight_decay
    
    def update_rule(self, state:dict, grads:ParamLike, params:ParamLike, **kwargs):
        grads_list, gtree = flatten_pytree(grads)
        params_list, ptree = flatten_pytree(params)
        state_list, stree = flatten_pytree(state) 
        state_updates = []
        updates = []

        i = 0 
        for g, p in zip(grads_list, params_list):
            _temp = None
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * p

            if self.momentum != 0.0:
                v = state["velocity"][i]
                v_new = self.momentum * v + g
                state_updates.append(v_new)

                if self.nestrov:
                    _update = g + self.momentum * v_new
                    updates.append(self.lr*_update)
                
                else:
                    _update = v_new
                    updates.append(self.lr*_update)

            else:
                v = state["velocity"][i]
                state_updates.append(v)
                _update = g
                updates.append(self.lr*_update)
            
            i += 1

        new_state = unflatten_pytree(state_updates, stree)
        updates_ = unflatten_pytree(updates, ptree)

        return new_state, updates_