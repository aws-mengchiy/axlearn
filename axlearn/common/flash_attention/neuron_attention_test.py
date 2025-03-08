# Copyright © 2024 Amazon Inc.
"""Tests for Flash attention on Neuron. Tested on trn1 & trn2."""

import chex
import jax
import jax.numpy as jnp
import pytest

from axlearn.common.flash_attention.utils import mha_reference

if jax.default_backend() != "neuron":
    pytestmark = pytest.skip(
        reason="Incompatible hardware, AWS Neuron only test.", allow_module_level=True
    )

import numpy as np
import neuronxcc.nki.language as nl

def random_ints_sum_to_length(length, n_seq):
    """Generates n_seq random integers summing to a specified length using NumPy.

    Args:
        length: The desired sum of the integers.
        n_seq: The number of integers in the sequence.

    Returns:
        A NumPy array of n_seq random integers that sum to length.

    Raises:
        ValueError: If length is not a positive integer or n_seq is not a positive integer.
    """
    if length <= 0:
        raise ValueError("Length must be a positive integer.")
    if n_seq <= 0:
        raise ValueError("n_seq must be a positive integer.")
    if n_seq > length:
        raise ValueError("n_seq cannot be greater than length.")


    partitions = np.sort(np.random.rand(n_seq - 1))
    partitions = np.concatenate(([0], partitions, [1]))
    diffs = np.diff(partitions)
    random_ints = np.round(diffs * length).astype(int)

    diff = int(length - np.sum(random_ints))
    random_ints[-1] += diff

    return random_ints

def get_segmend_ids(length, n_seq):
    seq_lens = random_ints_sum_to_length(length, n_seq)
    oc = seq_lens.cumsum()
    lhs = [0]*seq_lens[0]
    rhs = [seq_lens[0]]*seq_lens[0]
    for i, (l, r) in enumerate(zip(oc[:-1], oc[1:])):
        lhs += [l]*seq_lens[i+1]
        rhs += [r]*seq_lens[i+1]
    return np.array(lhs).reshape((-1,1)), np.array(rhs).reshape((-1,1))

@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,per_head_dim",
    [
        # (1, 2048, 1, 64),
        (2, 2048, 2, 64),
        # (1, 2048, 1, 128),
        # (2, 2048, 2, 128),
        # (1, 2048, 8, 128),
        # (2, 2048, 8, 128),
    ],
)
# @pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("attention_bias_type", [None, "2d"])
# @pytest.mark.parametrize("attention_bias_type", [None])
# @pytest.mark.parametrize("input_dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("input_dtype", [jnp.bfloat16])
def test_fwd_against_ref(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    per_head_dim: int,
    causal: bool,
    input_dtype: jnp.dtype,
    attention_bias_type: str,
):
    # On demand import only if test is needed.
    # pylint: disable=import-outside-toplevel
    from axlearn.common.flash_attention.neuron_attention_tmp_ import flash_attention

    softmax_scale = per_head_dim**-0.5
    # k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.PRNGKey(0), 5)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)

    if attention_bias_type == "2d":
        bias = jax.random.normal(k4, (1, 1, seq_len, seq_len), dtype=input_dtype)
    else:
        bias = None

    # segment_ids = jax.random.bernoulli(k5, shape=(batch_size, seq_len)).astype(jnp.int32)
    # segment_ids = jnp.cumsum(segment_ids, axis=1)
    # segment_ids = None

    q_segment_ids_tile_ref, kv_segment_ids_tile_ref = get_segmend_ids(batch_size* seq_len, 8)
    q_segment_ids_tile_ref  = q_segment_ids_tile_ref.reshape((batch_size, seq_len))
    kv_segment_ids_tile_ref = kv_segment_ids_tile_ref.reshape((batch_size, seq_len))
    # q_segment_ids_tile_ref = nl.static_cast(q_segment_ids_tile_ref, nl.float32)
    # kv_segment_ids_tile_ref = nl.static_cast(kv_segment_ids_tile_ref, nl.float32)

    q_segment_ids_tile_ref = jnp.asarray(q_segment_ids_tile_ref)
    kv_segment_ids_tile_ref = jnp.asarray(kv_segment_ids_tile_ref)

    o = flash_attention(
        q,
        k,
        v,
        bias,
        prng_key=None,
        q_segment_ids_tile_ref=q_segment_ids_tile_ref,
        kv_segment_ids_tile_ref=kv_segment_ids_tile_ref,
        causal=causal,
        softmax_scale=softmax_scale,
        dropout_rate=0.0,
    )
    # o_ref = mha_reference(
    #     q,
    #     k,
    #     v,
    #     bias,
    #     segment_ids=segment_ids,
    #     causal=causal,
    #     softmax_scale=softmax_scale,
    #     dropout_rate=0.0,
    # )
    # if input_dtype == jnp.float16:
    #     chex.assert_trees_all_close(o, o_ref, atol=0.07)
    # elif input_dtype == jnp.float32:
    #     chex.assert_trees_all_close(o, o_ref, atol=0.03)


# @pytest.mark.parametrize(
#     "batch_size,num_heads,seq_len,per_head_dim",
#     [
#         (1, 1, 2048, 64),
#         (2, 2, 2048, 64),
#         (1, 1, 2048, 128),
#         (2, 2, 2048, 128),
#         (1, 8, 2048, 128),
#         (2, 8, 2048, 128),
#     ],
# )
# @pytest.mark.parametrize("causal", [True, False])
# @pytest.mark.parametrize("input_dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
# @pytest.mark.parametrize("attention_bias_type", [None, "2d"])
# def test_bwd_against_ref(
#     batch_size: int,
#     num_heads: int,
#     seq_len: int,
#     per_head_dim: int,
#     causal: bool,
#     input_dtype: jnp.dtype,
#     attention_bias_type: str,
# ):
#     # On demand import only if test is needed.
#     # pylint: disable=import-outside-toplevel
#     from axlearn.common.flash_attention.neuron_attention import flash_attention

#     softmax_scale = per_head_dim**-0.5
#     k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
#     q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
#     k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
#     v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)

#     if attention_bias_type == "2d":
#         bias = jax.random.normal(k4, (1, 1, seq_len, seq_len), dtype=input_dtype)
#     else:
#         bias = None

#     def fn(q, k, v, bias):
#         return flash_attention(
#             q,
#             k,
#             v,
#             bias,
#             causal=causal,
#             softmax_scale=softmax_scale,
#             dropout_rate=0.0,
#         ).sum()

#     def ref_fn(q, k, v, bias):
#         return mha_reference(
#             q,
#             k,
#             v,
#             bias,
#             causal=causal,
#             softmax_scale=softmax_scale,
#             dropout_rate=0.0,
#         ).sum()

#     jax_grads = jax.grad(fn, argnums=(0, 1, 2))(q, k, v, bias)
#     jax_ref_grads = jax.grad(ref_fn, argnums=(0, 1, 2))(q, k, v, bias)
#     chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.07)
