# Tenx

*An Eager mode Dynamic Autodiff Library in active development*

numfire  provides JAX-like automatic differentiation with PyTree support but **Dynamic first**, built with a backend-agnostic architecture that automatically leverages available hardware.

## Quick Start

```python
import numfire.ndarray as nd
import numfire  as nf

# Basic autodiff - automatically uses GPU if available
def d2x(x):
    f = lambda x: x**2
    d = nf.grad(f)
    d2 = nf.grad(d) 
    return d(x), d2(x)

x = nd.array(2.)  # Automatically placed on best available device
print(d2x(x))  # (array(4.), array(2.))

# PyTree support with automatic device placement
def fun(x: dict):
    a = x['a']
    b = x['b']
    return a**b

# Structures are automatically moved to optimal device
x = {'a': nd.array(3.), 'b': nd.array(2.)}
print(nf.grad(fun)(x))
# {'a': array(6.), 'b': array(9.8875106)}
```

## Development Status

**Experimental Research Project**

numfire  is currently in **active research and development** as part of an academic exploration into minimal autodiff systems. This is not production-ready somtware, but rather a live prototype where we're actively experimenting with:

- Novel approaches to automatic differentiation
- Hardware-agnostic tensor abstractions
- PyTree-based gradient computation
- Dynamic backend selection strategies

**What this means for users:**
- Expect breaking API changes
- Performance characteristics are being actively measured and improved
- Not all NumPy operations are supported yet
- Documentation may lag behind implementation
- Some backends may be incomplete or experimental

**Current Development Focus:**
1. **Stabilizing the core autodiff engine** (95% complete)
2. **Expanding backend support** (CUDA stable via CuPy)
3. **Performance optimization** across different hardware
4. **API design** based on real-world use cases

**We invite collaboration from:**
- Researchers interested in autodiff systems
- Developers curious about backend abstraction
- Early adopters willing to experiment with bleeding-edge features

**Not yet recommended for:**
- Production systems
- Critical research (without thorough validation)
- Educational purposes (unless exploring autodiff internals)

This project represents the frontier of what's possible with minimal autodiff abstractions. Things will break, APIs will change, and we're learning as we build.

---

*numfire : An experiment in minimal, hardware-agnostic autodiff. Breaking things thoughtfully since 2025.*