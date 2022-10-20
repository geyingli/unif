""" GPT-2.
  Code revised from Open AI's implementation of GPT-2.
  See `https://github.com/openai/gpt-2`.
"""

import numpy as np

from ...third import tf
from .._base_._base_ import BaseEncoder, BaseDecoder
from .. import util


class GPT2(BaseDecoder, BaseEncoder):
    def __init__(self,
                 hparams,
                 is_training,
                 input_ids,
                 sample_weight=None,
                 scope="model",
                 given=1,
                 **kwargs):
        super().__init__()

        batch_size = util.get_shape_list(input_ids, expected_rank=2)[0]
        max_seq_length = hparams.n_predict

        with tf.variable_scope(scope):

            def _forward(input_ids, past=None):
                batch, sequence = shape_list(input_ids)

                wte = kwargs.get("tilda_embeddings")
                if wte is None:
                    wte = tf.get_variable(
                        "word_embeddings", [hparams.n_vocab, hparams.n_embed],
                        initializer=tf.random_normal_initializer(stddev=0.02))
                wpe = tf.get_variable(
                    "wpe", [hparams.n_ctx, hparams.n_embed],
                    initializer=tf.random_normal_initializer(stddev=0.01))
                past_length = 0 if past is None else tf.shape(past)[-2]
                h = (tf.gather(wte, input_ids) +
                     tf.gather(wpe, positions_for(input_ids, past_length)))

                # stacked transformer layers
                presents = []
                pasts = tf.unstack(past, axis=1) if past is not None else \
                    [None] * hparams.n_layer
                assert len(pasts) == hparams.n_layer
                for layer, past in enumerate(pasts):
                    h, present = block(
                        h, "h%d" % layer, past=past, hparams=hparams)
                    presents.append(present)
                present = tf.stack(presents, axis=1)
                h = norm(h, "ln_f")

                # Language model loss.  Do tokens <n predict token n?
                h_flat = tf.reshape(h, [batch*sequence, hparams.n_embed])
                logits = tf.matmul(h_flat, wte, transpose_b=True)
                logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])

                return logits, present

            # convert to labels
            label_ids = tf.concat(
                [input_ids[:, 1:],
                 tf.zeros([batch_size, 1], dtype=tf.int32)], axis=-1)

            # forward once
            if is_training:
                (logits, _) = _forward(input_ids)

                self.tensors["preds"] = tf.argmax(logits, axis=-1)

            # forward loop
            else:
                input_ids = input_ids[:, 0:given]

                for cur_length in range(given, max_seq_length + 1):
                    (logits, _) = _forward(input_ids)

                    pred_ids = tf.argmax(
                        logits[:, cur_length-1:cur_length, :], axis=-1)
                    pred_ids = tf.cast(pred_ids, tf.int32)
                    input_ids = tf.concat([input_ids, pred_ids], axis=-1)

                self.tensors["preds"] = input_ids

            # loss
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            one_hot_labels = tf.one_hot(label_ids, depth=hparams.n_vocab)
            per_token_loss = -tf.reduce_sum(
                one_hot_labels * log_probs, axis=-1)
            label_mask = tf.cast(tf.not_equal(label_ids, 0), tf.float32)
            per_example_loss = \
                tf.reduce_sum(per_token_loss * label_mask, axis=-1) / \
                tf.reduce_sum(label_mask, axis=-1)
            if sample_weight is not None:
                per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

            self.train_loss = tf.reduce_mean(per_example_loss)
            self.tensors["losses"] = per_example_loss


class GPT2Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            self.__setattr__(key, value)


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1]
        g = tf.get_variable("g", [n_state],
                            initializer=tf.constant_initializer(1))
        b = tf.get_variable("b", [n_state],
                            initializer=tf.constant_initializer(0))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable(
            "w", [1, nx, nf],
            initializer=tf.random_normal_initializer(stddev=w_init_stdev))
        b = tf.get_variable("b", [nf], initializer=tf.constant_initializer(0))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]),
                                 tf.reshape(w, [-1, nf])) + b, start + [nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn"t
      produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence,
                                      # features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads,
        # sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where
        # information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1], w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        a = tf.matmul(w, v)
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, "c_attn", n_state*3)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, "c_proj", n_state)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1]
        h = gelu(conv1d(x, "c_fc", n_state))
        h2 = conv1d(h, "c_proj", nx)
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1]
        a, present = attn(
            norm(x, "ln_1"), "attn", nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, "ln_2"), "mlp", nx*4, hparams=hparams)
        x = x + m
        return x, present


def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size,
            hparams.n_layer,
            2,
            hparams.n_head,
            sequence,
            hparams.n_embed // hparams.n_head]


def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name="value")
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)


def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


def get_decay_power(num_hidden_layers):
    decay_power = {
        "/word_embeddings": num_hidden_layers + 2,
        "/wpe": num_hidden_layers + 2,
        "ln_f/": 0,
    }
    for layer_idx in range(num_hidden_layers):
        decay_power["/h%d/" % layer_idx] = num_hidden_layers - layer_idx + 1
    return decay_power
