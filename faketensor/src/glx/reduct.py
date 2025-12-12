
def reshape_op(x, shape):
    if any(s is None for s in shape):
        known = [s for s in shape if s is not None]
        known_size = np.prod(known)
        total = x.size
        missing = total // known_size
        shape = [missing if s is None else s for s in shape]
    return np.reshape(x, shape)

