# -*- coding: utf-8 -*-

__all__ = ["ehrlich_aberth"]

from functools import partial

import numpy as np
from jax import numpy as jnp
from jax.lib import xla_client
from jax import core, dtypes, lax, jit
from jax.interpreters import ad, batching, xla
from jax.abstract_arrays import ShapedArray

# Register the CPU XLA custom calls
from . import cpu_ops

for _name, _value in cpu_ops.registrations().items():
    xla_client.register_cpu_custom_call_target(_name, _value)

# If the GPU version exists, also register those
try:
    from . import gpu_ops
except ImportError:
    gpu_ops = None
else:
    for _name, _value in gpu_ops.registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")
print("gpu_ops:", gpu_ops)

xops = xla_client.ops


# This function exposes the primitive to user code and this is the only
# public-facing function in this module
def ehrlich_aberth(coeffs):
    """Compute complex polynomial roots."""
    # Reshape to shape (size * (deg + 1),)
    coeffs = coeffs.reshape(-1, coeffs.shape[-1])

    # The C++ function expects the coefficients to be ordered as p[0] + p[1] * x + ...
    coeffs = coeffs[:, ::-1]

    return _ehrlich_aberth_prim.bind(coeffs)


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _ehrlich_aberth_abstract(coeffs):
    shape = coeffs.shape
    size = shape[0] # number of polynomials
    deg = shape[1] - 1 # degree of polynomials
    dtype = dtypes.canonicalize_dtype(coeffs.dtype)
    return ShapedArray((size * deg,), dtype)

# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _ehrlich_aberth_translation(c, coeffs, platform="cpu"):
    # The inputs have "shapes" that provide both the shape and the dtype
    coeffs_shape = c.get_shape(coeffs)

    # Input shapes
    dims_input = coeffs_shape.dimensions()

    size = dims_input[0] # number of polynomials
    deg = dims_input[1] - 1 # degree of polynomials

    # Output shapes
    dims_output = (size * deg,)

    # Extract the dtype
    dtype = coeffs_shape.element_type()
    assert coeffs_shape.element_type() == dtype
    assert coeffs_shape.dimensions() == dims_input

    shape_input = xla_client.Shape.array_shape(
        np.dtype(dtype), dims_input, tuple(range(len(dims_input) - 1, -1, -1))
    )
    shape_output = xla_client.Shape.array_shape(
        np.dtype(dtype), dims_output, tuple(range(len(dims_output) - 1, -1, -1))
    )

    if dtype == np.complex128:
        op_name = platform.encode() + b"_ehrlich_aberth"
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    # And then the following is what changes between the GPU and CPU
    if platform == "cpu":
        # On the CPU, we pass the size of the data as a the first input
        # argument
        return xops.CustomCallWithLayout(
            c,
            op_name,
            # The inputs:
            operands=(
                xops.ConstantLiteral(c, size),
                xops.ConstantLiteral(c, deg),
                coeffs,
            ),
            # The input shapes:
            operand_shapes_with_layout=(
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                shape_input,
            ),
            # The output shapes:
            shape_with_layout=shape_output,
        )
    
    elif platform == "gpu":
        if gpu_ops is None:
            raise ValueError(
                "The 'ehrlich_aberth_jax' module was not compiled with CUDA support"
            )

        # On the GPU, we do things a little differently and encapsulate the
        # dimension using the 'opaque' parameter
        opaque = gpu_ops.build_ehrlich_aberth_descriptor(size, deg)

        return xops.CustomCallWithLayout(
            c,
            op_name,
            operands=(coeffs,),
            operand_shapes_with_layout=(shape_input,),
            shape_with_layout=shape_output,
            opaque=opaque,
        )

    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


# **********************************
# *  SUPPORT FOR FORWARD AUTODIFF  *
# **********************************

# Here we define the differentiation rules using a JVP derived using implicit
# differentiation of Kepler's equation:
#
#  M = E - e * sin(E)
#  -> dM = dE * (1 - e * cos(E)) - de * sin(E)
#  -> dE/dM = 1 / (1 - e * cos(E))  and  de/dM = sin(E) / (1 - e * cos(E))
#
# In this case we don't need to define a transpose rule in order to support
# reverse and higher order differentiation. This might not be true in other
# applications, so check out the "How JAX primitives work" tutorial in the JAX
# documentation for more info as necessary.
# def _kepler_jvp(args, tangents):
#    mean_anom, ecc = args
#    d_mean_anom, d_ecc = tangents
#
#    # We use "bind" here because we don't want to mod the mean anomaly again
#    sin_ecc_anom, cos_ecc_anom = _kepler_prim.bind(mean_anom, ecc)
#
#    def zero_tangent(tan, val):
#        return lax.zeros_like_array(val) if type(tan) is ad.Zero else tan
#
#    # Propagate the derivatives
#    d_ecc_anom = (
#        zero_tangent(d_mean_anom, mean_anom) + zero_tangent(d_ecc, ecc) * sin_ecc_anom
#    ) / (1 - ecc * cos_ecc_anom)
#
#    return (
#        (sin_ecc_anom, cos_ecc_anom),
#        (cos_ecc_anom * d_ecc_anom, -sin_ecc_anom * d_ecc_anom,),
#    )
#

# ************************************
# *  SUPPORT FOR BATCHING WITH VMAP  *
# ************************************

# Our op already supports arbitrary dimensions so the batching rule is quite
# simple. The jax.lax.linalg module includes some example of more complicated
# batching rules if you need such a thing.
# def _kepler_batch(args, axes):
#    assert axes[0] == axes[1]
#    return ehrlich_aberth(*args), axes


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
_ehrlich_aberth_prim = core.Primitive("ehrlich_aberth")
_ehrlich_aberth_prim.def_impl(partial(xla.apply_primitive, _ehrlich_aberth_prim))
_ehrlich_aberth_prim.def_abstract_eval(_ehrlich_aberth_abstract)

# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["cpu"][_ehrlich_aberth_prim] = partial(
    _ehrlich_aberth_translation, platform="cpu"
)
xla.backend_specific_translations["gpu"][_ehrlich_aberth_prim] = partial(
   _ehrlich_aberth_translation, platform="gpu"
)

# Connect the JVP and batching rules
# ad.primitive_jvps[_ehrlich_aberth_prim] = _kepler_jvp
# batching.primitive_batchers[_ehrlich_aberth_prim] = _kepler_batch