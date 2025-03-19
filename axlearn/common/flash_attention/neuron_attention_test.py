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

def random_ints_sum_to_length(batch_size, length, n_seq):
    np.random.seed(420)

    """Generates batch_size sets of n_seq random integers summing to a specified length."""
    if length <= 0:
        raise ValueError("Length must be a positive integer.")
    if n_seq <= 0:
        raise ValueError("n_seq must be a positive integer.")
    if n_seq > length:
        raise ValueError("n_seq cannot be greater than length.")
    
    batch_partitions = np.sort(np.random.rand(batch_size, n_seq - 1), axis=-1)
    batch_partitions = np.concatenate([np.zeros((batch_size, 1)), batch_partitions, np.ones((batch_size, 1))], axis=-1)
    diffs = np.diff(batch_partitions, axis=-1)
    random_ints = np.round(diffs * length).astype(int)
    
    diff = length - np.sum(random_ints, axis=-1, keepdims=True)
    random_ints[:, -1] += diff.squeeze()
    
    return random_ints

def get_segment_ids(batch_size, length, n_seq):
    seq_lens = random_ints_sum_to_length(batch_size, length, n_seq)
    
    lhs_batch = []
    rhs_batch = []
    
    for b in range(batch_size):
        oc = seq_lens[b].cumsum()
        lhs = [0] * seq_lens[b][0]
        rhs = [seq_lens[b][0]] * seq_lens[b][0]
    segment_ids_batch = []
    
    for b in range(batch_size):
        oc = seq_lens[b].cumsum()
        lhs = [0] * seq_lens[b][0]
        rhs = [seq_lens[b][0]] * seq_lens[b][0]
        segment_ids = [0] * seq_lens[b][0]
        
        for i, (l, r) in enumerate(zip(oc[:-1], oc[1:])):
            lhs += [l] * seq_lens[b][i + 1]
            rhs += [r] * seq_lens[b][i + 1]
            segment_ids += [i + 1] * seq_lens[b][i + 1]
        
        lhs_batch.append(lhs)
        rhs_batch.append(rhs)
        segment_ids_batch.append(segment_ids)
    
    return seq_lens, np.array(lhs_batch), np.array(rhs_batch), np.array(segment_ids_batch)

@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,per_head_dim",
    [
        (1, 2048, 1, 64),
        (2, 2048, 2, 64),
        (1, 2048, 1, 128),
        (2, 2048, 2, 128),
        (1, 2048, 8, 128),
        (2, 2048, 8, 128),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
# @pytest.mark.parametrize("attention_bias_type", [None, "2d"])
@pytest.mark.parametrize("attention_bias_type", [None])
@pytest.mark.parametrize("input_dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
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
    from axlearn.common.flash_attention.neuron_attention import flash_attention

    softmax_scale = per_head_dim**-0.5
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(42), 4)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)

    if attention_bias_type == "2d":
        bias = jax.random.normal(k4, (1, 1, seq_len, seq_len), dtype=input_dtype)
    else:
        bias = None

    seq_lens, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, segment_ids_batch = get_segment_ids(batch_size, seq_len, 8)
    # q_segment_ids_tile_ref  = q_segment_ids_tile_ref.reshape((batch_size, seq_len))
    # kv_segment_ids_tile_ref = kv_segment_ids_tile_ref.reshape((batch_size, seq_len))
    q_segment_ids_tile_ref = nl.static_cast(q_segment_ids_tile_ref, nl.float32)
    kv_segment_ids_tile_ref = nl.static_cast(kv_segment_ids_tile_ref, nl.float32)

    q_segment_ids_tile_ref = jnp.asarray(q_segment_ids_tile_ref)
    kv_segment_ids_tile_ref = jnp.asarray(kv_segment_ids_tile_ref)
    segment_ids_batch = jnp.asarray(segment_ids_batch)

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
    o_ref = mha_reference(
        q,
        k,
        v,
        bias,
        segment_ids=segment_ids_batch,
        causal=causal,
        softmax_scale=softmax_scale,
        dropout_rate=0.0,
    )
    if input_dtype == jnp.float16:
        chex.assert_trees_all_close(o, o_ref, atol=0.07)
    elif input_dtype == jnp.bfloat16:
        chex.assert_trees_all_close(o, o_ref, atol=0.07)
    elif input_dtype == jnp.float32:
        chex.assert_trees_all_close(o, o_ref, atol=0.03)



@pytest.mark.parametrize(
    "batch_size,num_heads,seq_len,per_head_dim",
    [
        (1, 1, 2048, 64),
        (2, 2, 2048, 64),
        (1, 1, 2048, 128),
        (2, 2, 2048, 128),
        (1, 8, 2048, 128),
        (2, 8, 2048, 128),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("input_dtype", [jnp.bfloat16, jnp.float16, jnp.float32])
# @pytest.mark.parametrize("attention_bias_type", [None, "2d"])
@pytest.mark.parametrize("attention_bias_type", [None])
def test_bwd_against_ref(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    per_head_dim: int,
    causal: bool,
    input_dtype: jnp.dtype,
    attention_bias_type: str,
):
    # On demand import only if test is needed.
    # pylint: disable=import-outside-toplevel
    from axlearn.common.flash_attention.neuron_attention import flash_attention

    softmax_scale = per_head_dim**-0.5
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    q = jax.random.normal(k1, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    k = jax.random.normal(k2, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)
    v = jax.random.normal(k3, (batch_size, seq_len, num_heads, per_head_dim), dtype=input_dtype)

    if attention_bias_type == "2d":
        bias = jax.random.normal(k4, (1, 1, seq_len, seq_len), dtype=input_dtype)
    else:
        bias = None

    seq_lens, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, segment_ids_batch = get_segment_ids(batch_size, seq_len, 8)
    q_segment_ids_tile_ref = nl.static_cast(q_segment_ids_tile_ref, nl.float32)
    kv_segment_ids_tile_ref = nl.static_cast(kv_segment_ids_tile_ref, nl.float32)

    q_segment_ids_tile_ref = jnp.asarray(q_segment_ids_tile_ref)
    kv_segment_ids_tile_ref = jnp.asarray(kv_segment_ids_tile_ref)
    segment_ids_batch = jnp.asarray(segment_ids_batch)

    def fn(q, k, v, bias, q_segment_ids_tile_ref, kv_segment_ids_tile_ref):
        return flash_attention(
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
        ).sum()

    def ref_fn(q, k, v, bias, segment_ids_batch):
        return mha_reference(
            q,
            k,
            v,
            bias,
            segment_ids=segment_ids_batch,
            causal=causal,
            softmax_scale=softmax_scale,
            dropout_rate=0.0,
        ).sum()

    jax_grads = jax.grad(fn, argnums=(0, 1, 2))(q, k, v, bias, q_segment_ids_tile_ref, kv_segment_ids_tile_ref)
    jax_ref_grads = jax.grad(ref_fn, argnums=(0, 1, 2))(q, k, v, bias, segment_ids_batch)
    chex.assert_trees_all_close(jax_grads, jax_ref_grads, atol=0.07)
