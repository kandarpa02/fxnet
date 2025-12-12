from .runtime import xp
lib = xp()

OpRegister = {
    "Add": lambda x, y: lib.add(x, y), 
    "Mul": lambda x, y: lib.multiply(x, y), 
    # "Reshape": lambda x, shape: reshape_op(x, shape)
}
