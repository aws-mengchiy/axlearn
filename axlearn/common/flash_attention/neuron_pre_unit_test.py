# Copyright © 2024 Amazon Inc.
"""Tests for Flash attention on Neuron. Tested on trn1 & trn2."""
 
import chex
import jax
import jax.numpy as jnp
import pytest
from functools import partial
from axlearn.common.flash_attention.neuron_seq_packing_attention import nki_asm_get_sequence_bounds
  
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
    
    return seq_lens, np.array(lhs_batch), np.array(rhs_batch), np.array(segment_ids_batch)+1
 
@pytest.mark.parametrize(
    "batch_size,seq_len,num_heads,per_head_dim",
    [
        (1, 2048, 1, 64),
        # (2, 2048, 2, 64),
        # (1, 2048, 1, 128),
        # (2, 2048, 2, 128),
        # (1, 2048, 8, 128),
        # (2, 2048, 8, 128),
        # (1, 4096, 1, 64),
        # (2, 4096, 2, 64),
    ],
)
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("attention_bias_type", [None])
@pytest.mark.parametrize("input_dtype", [jnp.float32])
@pytest.mark.parametrize("seq_packing", [True])
def test_fwd_against_ref(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    per_head_dim: int,
    causal: bool,
    input_dtype: jnp.dtype,
    attention_bias_type: str,
    seq_packing: bool,
):
    # On demand import only if test is needed.
    # pylint: disable=import-outside-toplevel
 
    if seq_packing:
        q_segment_ids_tile_ref_pre, kv_segment_ids_tile_ref_pre, q_segment_ids_tile_ref, kv_segment_ids_tile_ref = preprocessing_wrapper(batch_size, seq_len)

        chex.assert_trees_all_close(q_segment_ids_tile_ref_pre, q_segment_ids_tile_ref, atol=0.0007)
        chex.assert_trees_all_close(kv_segment_ids_tile_ref_pre, kv_segment_ids_tile_ref, atol=0.0007)
 
    else:
        segment_ids_batch, q_segment_ids_tile_ref, kv_segment_ids_tile_ref = None, None, None

@partial(jax.jit, static_argnums=[0, 1])
def preprocessing_wrapper(batch_size, seq_len):

    seq_lens, q_segment_ids_tile_ref, kv_segment_ids_tile_ref, segment_ids_batch = get_segment_ids(batch_size, seq_len, 8)
    segment_ids_batch = jnp.asarray(segment_ids_batch)

    reshaped_segment_ids = segment_ids_batch[:, None, :]  # Add two singleton dimensions to [batch_size, 1, q_seq_len]
    reshaped_segment_ids = nl.static_cast(reshaped_segment_ids, nl.float32)
    partial_nki_asm_get_sequence_bounds = partial(nki_asm_get_sequence_bounds[(batch_size,)], output_tensor_dtype=nl.float32)
    processed_segment_ids = partial_nki_asm_get_sequence_bounds(reshaped_segment_ids)
    processed_segment_ids = jnp.asarray(processed_segment_ids)

    q_segment_ids_tile_ref_pre = processed_segment_ids[:, :, :seq_len].reshape((batch_size, seq_len))
    kv_segment_ids_tile_ref_pre = processed_segment_ids[:, :, -seq_len:].reshape((batch_size, seq_len))

    q_segment_ids_tile_ref_pre = jnp.asarray(q_segment_ids_tile_ref_pre)
    kv_segment_ids_tile_ref_pre = jnp.asarray(kv_segment_ids_tile_ref_pre)

    q_segment_ids_tile_ref = nl.static_cast(q_segment_ids_tile_ref, nl.float32)
    kv_segment_ids_tile_ref = nl.static_cast(kv_segment_ids_tile_ref, nl.float32)

    q_segment_ids_tile_ref = jnp.asarray(q_segment_ids_tile_ref)
    kv_segment_ids_tile_ref = jnp.asarray(kv_segment_ids_tile_ref)
    segment_ids_batch = jnp.asarray(segment_ids_batch)

    return q_segment_ids_tile_ref_pre, kv_segment_ids_tile_ref_pre, q_segment_ids_tile_ref, kv_segment_ids_tile_ref

