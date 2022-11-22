""" XLNet, an autoregressive permutation model.
  Code revised from XLNet team's implementation.
  See `https://github.com/zihangdai/xlnet`.
"""

import os
import json
import random
import numpy as np

from ...third import tf
from .._base_._base_ import BaseEncoder, BaseDecoder
from .. import util


SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4
special_symbols = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "<cls>": 3,
    "<sep>": 4,
    "<pad>": 5,
    "<mask>": 6,
    "<eod>": 7,
    "<eop>": 8,
}
UNK_ID = special_symbols["<unk>"]
CLS_ID = special_symbols["<cls>"]
SEP_ID = special_symbols["<sep>"]
MASK_ID = special_symbols["<mask>"]
EOD_ID = special_symbols["<eod>"]


class XLNetEncoder(BaseEncoder):
    """A wrapper of the XLNet model used during both pretraining and
    finetuning."""

    def __init__(self,
                 xlnet_config,
                 is_training,
                 input_ids,
                 seg_ids,
                 input_mask,
                 mems=None,
                 perm_mask=None,
                 target_mapping=None,
                 inp_q=None,
                 **kwargs):
        """
        Args:
          xlnet_config: XLNetConfig.
          is_training: bool, whether is training or not.
          input_ids: int32 Tensor in shape [len, bsz], the input token IDs.
          seg_ids: int32 Tensor in shape [len, bsz], the input segment IDs.
          input_mask: float32 Tensor in shape [len, bsz], the input mask.
              0 for real tokens and 1 for padding.
          mems: a list of float32 Tensors in shape [mem_len, bsz, d_model],
              memory from previous batches. The length of the list equals
              n_layer. If None, no memory is used.
          perm_mask: float32 Tensor in shape [len, len, bsz].
              If perm_mask[i, j, k] = 0, i attend to j in batch k;
              if perm_mask[i, j, k] = 1, i does not attend to j in batch k.
              If None, each position attends to all the others.
          target_mapping: float32 Tensor in shape [num_predict, len, bsz].
              If target_mapping[i, j, k] = 1, the i-th predict in batch k is
              on the j-th token.
              Only used during pretraining for partial prediction.
              Set to None during finetuning.
          inp_q: float32 Tensor in shape [len, bsz].
              1 for tokens with losses and 0 for tokens without losses.
              Only used during pretraining for two-stream attention.
              Set to None during finetuning.
        """

        run_config = XLNetRunConfig(
            is_training=is_training,
            bi_data=False,
            use_tpu=False,
            use_bfloat16=False,
            dropout=(0.1 if is_training else 0.0),
            dropatt=(0.1 if is_training else 0.0),
            init="normal",
            init_range=0.1,
            init_std=0.02,
            clamp_len=-1)
        initializer = _get_initializer(run_config)

        tfm_args = dict(
            n_token=xlnet_config.n_token,
            initializer=initializer,
            attn_type="bi",
            n_layer=xlnet_config.n_layer,
            d_model=xlnet_config.d_model,
            n_head=xlnet_config.n_head,
            d_head=xlnet_config.d_head,
            d_inner=xlnet_config.d_inner,
            ff_activation=xlnet_config.ff_activation,
            untie_r=xlnet_config.untie_r,

            is_training=run_config.is_training,
            use_bfloat16=run_config.use_bfloat16,
            use_tpu=run_config.use_tpu,
            dropout=run_config.dropout,
            dropatt=run_config.dropatt,

            mem_len=run_config.mem_len,
            reuse_len=run_config.reuse_len,
            bi_data=run_config.bi_data,
            clamp_len=run_config.clamp_len,
            same_length=run_config.same_length)

        input_args = dict(
            inp_k=input_ids,
            seg_id=seg_ids,
            input_mask=input_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            inp_q=inp_q,
            tilda_embeddings=kwargs.get("tilda_embeddings"))
        tfm_args.update(input_args)

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            (self.output, self.new_mems, self.lookup_table) = \
                transformer_xl(**tfm_args)

        self.input_mask = input_mask
        self.initializer = initializer
        self.xlnet_config = xlnet_config
        self.run_config = run_config

    def get_pooled_output(self):
        """
        Returns:
          float32 Tensor in shape [len, bsz, d_model]. The last layer hidden
          representation of XLNet.
        """
        # use "last" token as cls representation of sequence
        return self.output[-1]

    def get_sequence_output(self):
        """
        Returns:
          float32 Tensor in shape [len, bsz, d_model]. The last layer hidden
          representation of XLNet.
        """
        seq_length, batch_size, hidden_size = self.output.shape.as_list()
        return tf.transpose(self.output, [1, 0, 2])

    def get_new_memory(self):
        """
        Returns:
          list of float32 Tensors in shape [mem_len, bsz, d_model], the new
          memory that concatenates the previous memory with the current input
          representations.
          The length of the list equals n_layer.
        """
        return self.new_mems

    def get_embedding_table(self):
        """
        Returns:
          float32 Tensor in shape [n_token, d_model]. The embedding lookup
          table. Used for tying embeddings between input and output layers.
        """
        return self.lookup_table

    def get_initializer(self):
        """
        Returns:
          A tf initializer. Used to initialize variables in layers on top
              of XLNet.
        """
        return self.initializer


class XLNet(BaseDecoder):
    def __init__(self,
                 xlnet_config,
                 is_training,
                 input_ids,
                 seg_ids,
                 input_mask,
                 mems,
                 perm_mask,
                 target,
                 target_mask,
                 target_mapping,
                 inp_q,
                 sample_weight=None,
                 **kwargs):
        super().__init__()

        run_config = XLNetRunConfig(
            is_training=is_training,
            bi_data=True,
            use_tpu=False,
            use_bfloat16=False,
            dropout=(0.1 if is_training else 0.0),
            dropatt=(0.1 if is_training else 0.0),
            init="normal",
            init_range=0.1,
            init_std=0.02,
            clamp_len=-1)

        model = XLNetEncoder(
            xlnet_config=xlnet_config,
            is_training=is_training,
            input_ids=input_ids,
            seg_ids=seg_ids,
            input_mask=input_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            inp_q=inp_q,
            **kwargs)

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            per_example_loss, preds = lm_loss(
                hidden=model.get_sequence_output(),
                target=target,
                n_token=xlnet_config.n_token,
                d_model=xlnet_config.d_model,
                initializer=model.get_initializer(),
                lookup_table=model.get_embedding_table(),
                tie_weight=True,
                bi_data=run_config.bi_data,
                use_tpu=run_config.use_tpu,
                **kwargs)
            if sample_weight is not None:
                per_example_loss *= tf.expand_dims(sample_weight, axis=-1)

        self.train_loss = tf.reduce_sum(per_example_loss * target_mask) / tf.reduce_sum(target_mask)
        self.tensors["losses"] = per_example_loss * target_mask
        self.tensors["preds"] = preds
        self.tensors["mask"] = target_mask


class XLNetRunConfig:
    """XLNetRunConfig contains hyperparameters that could be different
    between pretraining and finetuning.
    These hyperparameters can also be changed from run to run.
    We store them separately from XLNetConfig for flexibility.
    """

    def __init__(self,
                 is_training, use_tpu, use_bfloat16, dropout, dropatt,
                 init="normal", init_range=0.1, init_std=0.02, mem_len=None,
                 reuse_len=None, bi_data=True, clamp_len=-1,
                 same_length=False):
        """
        Args:
          is_training: bool, whether in training mode.
          use_tpu: bool, whether TPUs are used.
          use_bfloat16: bool, use bfloat16 instead of float32.
          dropout: float, dropout rate.
          dropatt: float, dropout rate on attention probabilities.
          init: str, the initialization scheme, either "normal" or "uniform".
          init_range: float, initialize the parameters with a uniform
              distribution in [-init_range, init_range]. Only effective
              when init="uniform".
          init_std: float, initialize the parameters with a normal
              distribution with mean 0 and stddev init_std. Only effective
              when init="normal".
          mem_len: int, the number of tokens to cache.
          reuse_len: int, the number of tokens in the currect batch to be
              cached and reused in the future.
          bi_data: bool, whether to use bidirectional input pipeline.
              Usually set to True during pretraining and False during
              finetuning.
          clamp_len: int, clamp all relative distances larger than clamp_len.
              -1 means no clamping.
          same_length: bool, whether to use the same attention length for
              each token.
        """

        self.init = init
        self.init_range = init_range
        self.init_std = init_std
        self.is_training = is_training
        self.dropout = dropout
        self.dropatt = dropatt
        self.use_tpu = use_tpu
        self.use_bfloat16 = use_bfloat16
        self.mem_len = mem_len
        self.reuse_len = reuse_len
        self.bi_data = bi_data
        self.clamp_len = clamp_len
        self.same_length = same_length


def _get_initializer(FLAGS):
    """Get variable intializer."""
    if FLAGS.init == "uniform":
        initializer = tf.initializers.random_uniform(
            minval=-FLAGS.init_range,
            maxval=FLAGS.init_range,
            seed=None)
    elif FLAGS.init == "normal":
        initializer = tf.initializers.random_normal(
            stddev=FLAGS.init_std,
            seed=None)
    else:
        raise ValueError("Initializer {} not supported".format(FLAGS.init))
    return initializer


def embedding_lookup(x, n_token, d_embed, initializer, use_tpu=True,
                     scope="embedding", tilda_embeddings=None,
                     reuse=None, dtype=tf.float32):
    """TPU and GPU embedding_lookup function."""
    if tilda_embeddings is not None:
      lookup_table = tilda_embeddings
    else:
      with tf.variable_scope(scope, reuse=reuse):
          lookup_table = tf.get_variable(
              "lookup_table", [n_token, d_embed], dtype=dtype,
              initializer=initializer)
    if use_tpu:
        one_hot_idx = tf.one_hot(x, n_token, dtype=dtype)
        if one_hot_idx.shape.ndims == 2:
            return (tf.einsum("in,nd->id", one_hot_idx, lookup_table),
                    lookup_table)
        else:
            return (tf.einsum("ibn,nd->ibd", one_hot_idx, lookup_table),
                    lookup_table)
    else:
        return tf.nn.embedding_lookup(lookup_table, x), lookup_table


def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum("i,d->id", pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    pos_emb = pos_emb[:, None, :]

    if bsz is not None:
        pos_emb = tf.tile(pos_emb, [1, bsz, 1])

    return pos_emb


def positionwise_ffn(inp, d_model, d_inner, dropout, kernel_initializer,
                     activation_type="relu", scope="ff", is_training=True,
                     reuse=None):
    """Position-wise Feed-forward Network."""
    if activation_type == "relu":
        activation = tf.nn.relu
    elif activation_type == "gelu":
        activation = util.gelu
    else:
        raise ValueError("Unsupported activation type %s" % activation_type)

    output = inp
    with tf.variable_scope(scope, reuse=reuse):
        output = tf.layers.dense(
            output, d_inner, activation=activation,
            kernel_initializer=kernel_initializer,
            name="layer_1")
        output = tf.layers.dropout(
            output, dropout, training=is_training,
            name="drop_1")
        output = tf.layers.dense(
            output, d_model,
            kernel_initializer=kernel_initializer,
            name="layer_2")
        output = tf.layers.dropout(
            output, dropout, training=is_training,
            name="drop_2")
        output = util.layer_norm(
            output + inp,
            scope="LayerNorm")
    return output


def head_projection(h, d_model, n_head, d_head, kernel_initializer, name):
    """Project hidden states to a specific head with a 4D-shape."""
    proj_weight = tf.get_variable(
        "{}/kernel".format(name),
        [d_model, n_head, d_head], dtype=h.dtype,
        initializer=kernel_initializer)
    head = tf.einsum("ibh,hnd->ibnd", h, proj_weight)

    return head


def post_attention(h, attn_vec, d_model, n_head, d_head, dropout, is_training,
                   kernel_initializer, residual=True):
    """Post-attention processing."""
    # post-attention projection (back to `d_model`)
    proj_o = tf.get_variable(
        "o/kernel", [d_model, n_head, d_head],
        dtype=h.dtype, initializer=kernel_initializer)
    attn_out = tf.einsum("ibnd,hnd->ibh", attn_vec, proj_o)

    attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
    if residual:
        output = util.layer_norm(attn_out + h)
    else:
        output = util.L(attn_out)

    return output


def abs_attn_core(q_head, k_head, v_head, attn_mask, dropatt, is_training,
                  scale):
    """Core absolute positional attention operations."""

    attn_score = tf.einsum("ibnd,jbnd->ijbn", q_head, k_head)
    attn_score *= scale
    if attn_mask is not None:
        attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

    # attention output
    attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, v_head)

    return attn_vec


def rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                  r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt,
                  is_training, scale):
    """Core relative positional attention operations."""

    # content based attention score
    ac = tf.einsum("ibnd,jbnd->ijbn", q_head + r_w_bias, k_head_h)

    # position based attention score
    bd = tf.einsum("ibnd,jbnd->ijbn", q_head + r_r_bias, k_head_r)
    bd = rel_shift(bd, klen=tf.shape(ac)[1])

    # segment based attention score
    if seg_mat is None:
        ef = 0
    else:
        ef = tf.einsum("ibnd,snd->ibns", q_head + r_s_bias, seg_embed)
        ef = tf.einsum("ijbs,ibns->ijbn", seg_mat, ef)

    # merge attention scores and perform masking
    attn_score = (ac + bd + ef) * scale
    if attn_mask is not None:
        # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
        attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = tf.layers.dropout(attn_prob, dropatt, training=is_training)

    # attention output
    attn_vec = tf.einsum("ijbn,jbnd->ibnd", attn_prob, v_head_h)

    return attn_vec


def rel_shift(x, klen=-1):
    """perform relative shift to form the relative attention score."""
    x_size = tf.shape(x)

    x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
    x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

    return x


def _create_mask(qlen, mlen, dtype=tf.float32, same_length=False):
    """create causal attention mask."""
    attn_mask = tf.ones([qlen, qlen], dtype=dtype)
    mask_u = tf.matrix_band_part(attn_mask, 0, -1)
    mask_dia = tf.matrix_band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen], dtype=dtype)
    ret = tf.concat([attn_mask_pad, mask_u - mask_dia], 1)
    if same_length:
        mask_l = tf.matrix_band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)

    return ret


def _cache_mem(curr_out, prev_mem, mem_len, reuse_len=None):
    """cache hidden states into memory."""
    if mem_len is None or mem_len == 0:
        return None
    else:
        if reuse_len is not None and reuse_len > 0:
            curr_out = curr_out[:reuse_len]

        if prev_mem is None:
            new_mem = curr_out[-mem_len:]
        else:
            new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]

    return tf.stop_gradient(new_mem)


def relative_positional_encoding(qlen, klen, d_model, clamp_len, attn_type,
                                 bi_data, bsz=None, dtype=None):
    """create relative positional encoding."""
    freq_seq = tf.range(0, d_model, 2.0)
    if dtype is not None and dtype != tf.float32:
        freq_seq = tf.cast(freq_seq, dtype=dtype)
    inv_freq = 1 / (10000 ** (freq_seq / d_model))

    if attn_type == "bi":
        # beg, end = klen - 1, -qlen
        beg, end = klen, -qlen
    elif attn_type == "uni":
        # beg, end = klen - 1, -1
        beg, end = klen, -1
    else:
        raise ValueError("Unknown `attn_type` {}.".format(attn_type))

    if bi_data:
        fwd_pos_seq = tf.range(beg, end, -1.0)
        bwd_pos_seq = tf.range(-beg, -end, 1.0)

        if dtype is not None and dtype != tf.float32:
            fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
            bwd_pos_seq = tf.cast(bwd_pos_seq, dtype=dtype)

        if clamp_len > 0:
            fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
            bwd_pos_seq = tf.clip_by_value(bwd_pos_seq, -clamp_len, clamp_len)

        if bsz is not None:
            # With bi_data, the batch size should be divisible by 2.
            assert bsz%2 == 0
            fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz//2)
            bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq, bsz//2)
        else:
            fwd_pos_emb = positional_embedding(fwd_pos_seq, inv_freq)
            bwd_pos_emb = positional_embedding(bwd_pos_seq, inv_freq)

        pos_emb = tf.concat([fwd_pos_emb, bwd_pos_emb], axis=1)
    else:
        fwd_pos_seq = tf.range(beg, end, -1.0)
        if dtype is not None and dtype != tf.float32:
            fwd_pos_seq = tf.cast(fwd_pos_seq, dtype=dtype)
        if clamp_len > 0:
            fwd_pos_seq = tf.clip_by_value(fwd_pos_seq, -clamp_len, clamp_len)
        pos_emb = positional_embedding(fwd_pos_seq, inv_freq, bsz)

    return pos_emb


def multihead_attn(q, k, v, attn_mask, d_model, n_head, d_head, dropout,
                   dropatt, is_training, kernel_initializer, residual=True,
                   scope="abs_attn", reuse=None):
    """Standard multi-head attention with absolute positional embedding."""

    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope, reuse=reuse):
        # attention heads
        q_head = head_projection(
            q, d_model, n_head, d_head, kernel_initializer, "q")
        k_head = head_projection(
            k, d_model, n_head, d_head, kernel_initializer, "k")
        v_head = head_projection(
            v, d_model, n_head, d_head, kernel_initializer, "v")

        # attention vector
        attn_vec = abs_attn_core(
            q_head, k_head, v_head, attn_mask, dropatt,
            is_training, scale)

        # post processing
        output = post_attention(
            v, attn_vec, d_model, n_head, d_head, dropout,
            is_training, kernel_initializer, residual)

    return output


def rel_multihead_attn(h, r, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed,
                       attn_mask, mems, d_model, n_head, d_head, dropout,
                       dropatt, is_training, kernel_initializer,
                       scope="rel_attn", reuse=None):
    """Multi-head attention with relative positional encoding."""

    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope, reuse=reuse):
        if mems is not None and mems.shape.ndims > 1:
            cat = tf.concat([mems, h], 0)
        else:
            cat = h

        # content heads
        q_head_h = head_projection(
            h, d_model, n_head, d_head, kernel_initializer, "q")
        k_head_h = head_projection(
            cat, d_model, n_head, d_head, kernel_initializer, "k")
        v_head_h = head_projection(
            cat, d_model, n_head, d_head, kernel_initializer, "v")

        # positional heads
        k_head_r = head_projection(
            r, d_model, n_head, d_head, kernel_initializer, "r")

        # core attention ops
        attn_vec = rel_attn_core(
            q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
            r_w_bias, r_r_bias, r_s_bias,
            attn_mask, dropatt, is_training, scale)

        # post processing
        output = post_attention(
            h, attn_vec, d_model, n_head, d_head, dropout,
            is_training, kernel_initializer)

    return output


def two_stream_rel_attn(h, g, r, mems, r_w_bias, r_r_bias, seg_mat, r_s_bias,
                        seg_embed, attn_mask_h, attn_mask_g, target_mapping,
                        d_model, n_head, d_head, dropout, dropatt, is_training,
                        kernel_initializer, scope="rel_attn"):
    """Two-stream attention with relative positional encoding."""

    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope, reuse=False):

        # content based attention score
        if mems is not None and mems.shape.ndims > 1:
            cat = tf.concat([mems, h], 0)
        else:
            cat = h

        # content-based key head
        k_head_h = head_projection(
            cat, d_model, n_head, d_head, kernel_initializer, "k")

        # content-based value head
        v_head_h = head_projection(
            cat, d_model, n_head, d_head, kernel_initializer, "v")

        # position-based key head
        k_head_r = head_projection(
            r, d_model, n_head, d_head, kernel_initializer, "r")

        ##### h-stream
        # content-stream query head
        q_head_h = head_projection(
            h, d_model, n_head, d_head, kernel_initializer, "q")

        # core attention ops
        attn_vec_h = rel_attn_core(
            q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
            r_w_bias, r_r_bias, r_s_bias,
            attn_mask_h, dropatt, is_training, scale)

        # post processing
        output_h = post_attention(
            h, attn_vec_h, d_model, n_head, d_head, dropout,
            is_training, kernel_initializer)

    with tf.variable_scope(scope, reuse=True):
        ##### g-stream
        # query-stream query head
        q_head_g = head_projection(
            g, d_model, n_head, d_head, kernel_initializer, "q")

        # core attention ops
        if target_mapping is not None:
            q_head_g = tf.einsum("mbnd,mlb->lbnd", q_head_g, target_mapping)
            attn_vec_g = rel_attn_core(
                q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                r_w_bias, r_r_bias, r_s_bias,
                attn_mask_g, dropatt, is_training, scale)
            attn_vec_g = tf.einsum(
                "lbnd,mlb->mbnd", attn_vec_g, target_mapping)
        else:
            attn_vec_g = rel_attn_core(
                q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                r_w_bias, r_r_bias, r_s_bias,
                attn_mask_g, dropatt, is_training, scale)

        # post processing
        output_g = post_attention(
            g, attn_vec_g, d_model, n_head, d_head, dropout,
            is_training, kernel_initializer)

        return output_h, output_g


def transformer_xl(inp_k, n_token, n_layer, d_model, n_head,
                   d_head, d_inner, dropout, dropatt, attn_type,
                   bi_data, initializer, is_training, mem_len=None,
                   inp_q=None, mems=None,
                   same_length=False, clamp_len=-1, untie_r=False,
                   use_tpu=True, input_mask=None,
                   perm_mask=None, seg_id=None, reuse_len=None,
                   ff_activation="relu", target_mapping=None,
                   use_bfloat16=False, scope="transformer",
                   tilda_embeddings=None, **kwargs):
    """
    Defines a Transformer-XL computation graph with additional
    support for XLNet.

      Args:

      inp_k: int32 Tensor in shape [len, bsz], the input token IDs.
      seg_id: int32 Tensor in shape [len, bsz], the input segment IDs.
      input_mask: float32 Tensor in shape [len, bsz], the input mask.
          0 for real tokens and 1 for padding.
      mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
          from previous batches. The length of the list equals n_layer.
          If None, no memory is used.
      perm_mask: float32 Tensor in shape [len, len, bsz].
          If perm_mask[i, j, k] = 0, i attend to j in batch k;
          if perm_mask[i, j, k] = 1, i does not attend to j in batch k.
          If None, each position attends to all the others.
      target_mapping: float32 Tensor in shape [num_predict, len, bsz].
          If target_mapping[i, j, k] = 1, the i-th predict in batch k is
          on the j-th token.
          Only used during pretraining for partial prediction.
          Set to None during finetuning.
      inp_q: float32 Tensor in shape [len, bsz].
          1 for tokens with losses and 0 for tokens without losses.
          Only used during pretraining for two-stream attention.
          Set to None during finetuning.

      n_layer: int, the number of layers.
      d_model: int, the hidden size.
      n_head: int, the number of attention heads.
      d_head: int, the dimension size of each attention head.
      d_inner: int, the hidden size in feed-forward layers.
      ff_activation: str, "relu" or "gelu".
      untie_r: bool, whether to untie the biases in attention.
      n_token: int, the vocab size.

      is_training: bool, whether in training mode.
      use_tpu: bool, whether TPUs are used.
      use_bfloat16: bool, use bfloat16 instead of float32.
      dropout: float, dropout rate.
      dropatt: float, dropout rate on attention probabilities.
      init: str, the initialization scheme, either "normal" or "uniform".
      init_range: float, initialize the parameters with a uniform distribution
          in [-init_range, init_range]. Only effective when init="uniform".
      init_std: float, initialize the parameters with a normal distribution
          with mean 0 and stddev init_std. Only effective when init="normal".
      mem_len: int, the number of tokens to cache.
      reuse_len: int, the number of tokens in the currect batch to be cached
          and reused in the future.
      bi_data: bool, whether to use bidirectional input pipeline.
          Usually set to True during pretraining and False during finetuning.
      clamp_len: int, clamp all relative distances larger than clamp_len.
          -1 means no clamping.
      same_length: bool, whether to use the same attention length for each token.
      summary_type: str, "last", "first", "mean", or "attn". The method
          to pool the input to get a vector representation.
      initializer: A tf initializer.
      scope: scope name for the computation graph.
    """
    tf_float = tf.bfloat16 if use_bfloat16 else tf.float32

    new_mems = []
    with tf.variable_scope(scope):
        if untie_r:
            r_w_bias = tf.get_variable("r_w_bias", [n_layer, n_head, d_head],
                                       dtype=tf_float, initializer=initializer)
            r_r_bias = tf.get_variable("r_r_bias", [n_layer, n_head, d_head],
                                       dtype=tf_float, initializer=initializer)
        else:
            r_w_bias = tf.get_variable("r_w_bias", [n_head, d_head],
                                       dtype=tf_float, initializer=initializer)
            r_r_bias = tf.get_variable("r_r_bias", [n_head, d_head],
                                       dtype=tf_float, initializer=initializer)

        bsz = tf.shape(inp_k)[1]
        qlen = tf.shape(inp_k)[0]
        mlen = tf.shape(mems[0])[0] if mems is not None else 0
        klen = mlen + qlen

        ##### Attention mask
        # causal attention mask
        if attn_type == "uni":
            attn_mask = _create_mask(qlen, mlen, tf_float, same_length)
            attn_mask = attn_mask[:, :, None, None]
        elif attn_type == "bi":
            attn_mask = None
        else:
            raise ValueError("Unsupported attention type: %s" % attn_type)

        # data mask: input mask & perm mask
        if input_mask is not None and perm_mask is not None:
            data_mask = input_mask[None] + perm_mask
        elif input_mask is not None and perm_mask is None:
            data_mask = input_mask[None]
        elif input_mask is None and perm_mask is not None:
            data_mask = perm_mask
        else:
            data_mask = None

        if data_mask is not None:
            # all mems can be attended to
            mems_mask = tf.zeros([tf.shape(data_mask)[0], mlen, bsz],
                                 dtype=tf_float)
            data_mask = tf.cast(data_mask, dtype=tf.float32)
            data_mask = tf.concat([mems_mask, data_mask], 1)
            if attn_mask is None:
                attn_mask = data_mask[:, :, :, None]
            else:
                attn_mask += data_mask[:, :, :, None]

        if attn_mask is not None:
            attn_mask = tf.cast(attn_mask > 0, dtype=tf_float)

        if attn_mask is not None:
            non_tgt_mask = -tf.eye(qlen, dtype=tf_float)
            non_tgt_mask = tf.concat([tf.zeros([qlen, mlen], dtype=tf_float),
                                      non_tgt_mask], axis=-1)
            non_tgt_mask = tf.cast(
                (attn_mask + non_tgt_mask[:, :, None, None]) > 0,
                dtype=tf_float)
        else:
            non_tgt_mask = None

        ##### Word embedding
        word_emb_k, lookup_table = embedding_lookup(
            x=inp_k,
            n_token=n_token,
            d_embed=d_model,
            initializer=initializer,
            use_tpu=use_tpu,
            dtype=tf_float,
            scope="word_embedding",
            tilda_embeddings=tilda_embeddings)

        if inp_q is not None:
            with tf.variable_scope("mask_emb"):
                mask_emb = tf.get_variable(
                    "mask_emb", [1, 1, d_model], dtype=tf_float)
                if target_mapping is not None:
                    word_emb_q = tf.tile(
                        mask_emb, [tf.shape(target_mapping)[0], bsz, 1])
                else:
                    inp_q_ext = inp_q[:, :, None]
                    word_emb_q = \
                        inp_q_ext * mask_emb + (1 - inp_q_ext) * word_emb_k
        output_h = tf.layers.dropout(
            word_emb_k, dropout, training=is_training)
        if inp_q is not None:
            output_g = tf.layers.dropout(
                word_emb_q, dropout, training=is_training)

        ##### Segment embedding
        if seg_id is not None:
            if untie_r:
                r_s_bias = tf.get_variable(
                    "r_s_bias", [n_layer, n_head, d_head],
                    dtype=tf_float, initializer=initializer)
            else:
                # default case (tie)
                r_s_bias = tf.get_variable(
                    "r_s_bias", [n_head, d_head],
                    dtype=tf_float, initializer=initializer)

            seg_embed = tf.get_variable(
                "seg_embed", [n_layer, 2, n_head, d_head],
                dtype=tf_float, initializer=initializer)

            # Convert `seg_id` to one-hot `seg_mat`
            mem_pad = tf.zeros([mlen, bsz], dtype=tf.int32)
            cat_ids = tf.concat([mem_pad, seg_id], 0)

            # `1` indicates not in the same segment [qlen x klen x bsz]
            seg_mat = tf.cast(
                tf.logical_not(tf.equal(seg_id[:, None], cat_ids[None, :])),
                tf.int32)
            seg_mat = tf.one_hot(seg_mat, 2, dtype=tf_float)
        else:
            seg_mat = None

        ##### Positional encoding
        pos_emb = relative_positional_encoding(
            qlen, klen, d_model, clamp_len, attn_type, bi_data,
            bsz=bsz, dtype=tf_float)
        pos_emb = tf.layers.dropout(pos_emb, dropout, training=is_training)

        ##### Attention layers
        if mems is None:
            mems = [None] * n_layer

        for i in range(n_layer):
            # cache new mems
            new_mems.append(_cache_mem(output_h, mems[i], mem_len, reuse_len))

            # segment bias
            if seg_id is None:
                r_s_bias_i = None
                seg_embed_i = None
            else:
                r_s_bias_i = r_s_bias if not untie_r else r_s_bias[i]
                seg_embed_i = seg_embed[i]

            with tf.variable_scope("layer_{}".format(i)):
                if inp_q is not None:
                    output_h, output_g = two_stream_rel_attn(
                        h=output_h,
                        g=output_g,
                        r=pos_emb,
                        r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                        r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                        seg_mat=seg_mat,
                        r_s_bias=r_s_bias_i,
                        seg_embed=seg_embed_i,
                        attn_mask_h=non_tgt_mask,
                        attn_mask_g=attn_mask,
                        mems=mems[i],
                        target_mapping=target_mapping,
                        d_model=d_model,
                        n_head=n_head,
                        d_head=d_head,
                        dropout=dropout,
                        dropatt=dropatt,
                        is_training=is_training,
                        kernel_initializer=initializer)
                    reuse = True
                else:
                    reuse = False

                    output_h = rel_multihead_attn(
                        h=output_h,
                        r=pos_emb,
                        r_w_bias=r_w_bias if not untie_r else r_w_bias[i],
                        r_r_bias=r_r_bias if not untie_r else r_r_bias[i],
                        seg_mat=seg_mat,
                        r_s_bias=r_s_bias_i,
                        seg_embed=seg_embed_i,
                        attn_mask=non_tgt_mask,
                        mems=mems[i],
                        d_model=d_model,
                        n_head=n_head,
                        d_head=d_head,
                        dropout=dropout,
                        dropatt=dropatt,
                        is_training=is_training,
                        kernel_initializer=initializer,
                        reuse=reuse)

                if inp_q is not None:
                    output_g = positionwise_ffn(
                        inp=output_g,
                        d_model=d_model,
                        d_inner=d_inner,
                        dropout=dropout,
                        kernel_initializer=initializer,
                        activation_type=ff_activation,
                        is_training=is_training)

                output_h = positionwise_ffn(
                    inp=output_h,
                    d_model=d_model,
                    d_inner=d_inner,
                    dropout=dropout,
                    kernel_initializer=initializer,
                    activation_type=ff_activation,
                    is_training=is_training,
                    reuse=reuse)

        if inp_q is not None:
            output = tf.layers.dropout(output_g, dropout, training=is_training)
        else:
            output = tf.layers.dropout(output_h, dropout, training=is_training)

        return output, new_mems, lookup_table


def lm_loss(hidden, target, n_token, d_model, initializer, lookup_table=None,
            tie_weight=False, bi_data=True, use_tpu=False, **kwargs):
    """doc."""

    with tf.variable_scope("lm_loss"):
        if tie_weight:
            assert lookup_table is not None, \
                "lookup_table cannot be None for tie_weight"
            softmax_w = lookup_table
        else:
            softmax_w = tf.get_variable(
                "weight", [n_token, d_model],
                dtype=hidden.dtype, initializer=initializer)

        softmax_b = tf.get_variable(
            "bias", [n_token], dtype=hidden.dtype,
            initializer=tf.zeros_initializer())

        logits = tf.einsum("ibd,nd->ibn", hidden, softmax_w) + softmax_b
        preds = tf.argmax(logits, axis=-1)

        if use_tpu:
            one_hot_target = tf.one_hot(target, n_token, dtype=logits.dtype)
            loss = -tf.reduce_sum(
                tf.nn.log_softmax(logits) * one_hot_target, -1)
        else:
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target, logits=logits)

        return loss, preds


def summarize_sequence(summary_type, hidden, d_model, n_head, d_head, dropout,
                       dropatt, input_mask, is_training, initializer,
                       scope=None, reuse=None, use_proj=True):

    """
    Different classification tasks may not may not share the same parameters
    to summarize the sequence features.

    If shared, one can keep the `scope` to the default value `None`.
    Otherwise, one should specify a different `scope` for each task.
    """

    with tf.variable_scope(scope, "sequnece_summary", reuse=reuse):
        if summary_type == "last":
            summary = hidden[-1]
        elif summary_type == "first":
            summary = hidden[0]
        elif summary_type == "mean":
            summary = tf.reduce_mean(hidden, axis=0)
        elif summary_type == "attn":
            bsz = tf.shape(hidden)[1]

            summary_bias = tf.get_variable("summary_bias", [d_model],
                                           dtype=hidden.dtype,
                                           initializer=initializer)
            summary_bias = tf.tile(summary_bias[None, None], [1, bsz, 1])

            if input_mask is not None:
                input_mask = input_mask[None, :, :, None]

            summary = multihead_attn(
                summary_bias, hidden, hidden, input_mask,
                d_model, n_head, d_head, dropout, dropatt,
                is_training, initializer, residual=False)
            summary = summary[0]
        else:
            raise ValueError("Unsupported summary type %s" % summary_type)

        # use another projection as in BERT
        if use_proj:
            summary = tf.layers.dense(
                summary,
                d_model,
                activation=tf.tanh,
                kernel_initializer=initializer,
                name="summary")

        # dropout
        summary = tf.layers.dropout(
            summary, dropout, training=is_training,
            name="dropout")

    return summary


def classification_loss(hidden, labels, n_class, initializer, scope,
                        reuse=None, return_hidden=False):
    """
    Different classification tasks should use different scope names to ensure
    different dense layers (parameters) are used to produce the logits.

    An exception will be in transfer learning, where one hopes to transfer
    the classification weights.
    """

    with tf.variable_scope(scope, reuse=reuse):
        logits = tf.layers.dense(
            hidden,
            n_class,
            kernel_initializer=initializer,
            name="logit")

        one_hot_target = tf.one_hot(labels, n_class, dtype=hidden.dtype)
        loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * one_hot_target, -1)

        if return_hidden:
            return loss, logits

        return loss


def regression_loss(hidden, labels, initializer, scope, reuse=None,
                    return_hidden=False):
    with tf.variable_scope(scope, reuse=reuse):
        logits = tf.layers.dense(
            hidden,
            1,
            kernel_initializer=initializer,
            name="logit")

        logits = tf.squeeze(logits, axis=-1)
        loss = tf.square(logits - labels)

        if return_hidden:
            return loss, logits

        return loss


class XLNetConfig:
    """XLNetConfig contains hyperparameters that are specific to a model
    checkpoint; i.e., these hyperparameters should be the same between
    pretraining and finetuning.

    The following hyperparameters are defined:
      n_layer: int, the number of layers.
      d_model: int, the hidden size.
      n_head: int, the number of attention heads.
      d_head: int, the dimension size of each attention head.
      d_inner: int, the hidden size in feed-forward layers.
      ff_activation: str, "relu" or "gelu".
      untie_r: bool, whether to untie the biases in attention.
      n_token: int, the vocab size.
    """

    def __init__(self, FLAGS=None, json_path=None):
        """Constructing an XLNetConfig.
        One of FLAGS or json_path should be provided."""

        assert FLAGS is not None or json_path is not None

        self.keys = ["n_layer", "d_model", "n_head", "d_head", "d_inner",
                     "ff_activation", "untie_r", "n_token"]

        if FLAGS is not None:
            self.init_from_flags(FLAGS)

        if json_path is not None:
            self.init_from_json(json_path)

    def init_from_flags(self, FLAGS):
        for key in self.keys:
            setattr(self, key, getattr(FLAGS, key))

    def init_from_json(self, json_path):
        with tf.gfile.Open(json_path) as f:
            json_data = json.load(f)
            for key in json_data:
                setattr(self, key, json_data[key])

    def to_json(self, json_path):
        """Save XLNetConfig to a json file."""
        json_data = {}
        for key in self.keys:
            json_data[key] = getattr(self, key)

        json_dir = os.path.dirname(json_path)
        if not tf.gfile.Exists(json_dir):
            tf.gfile.MakeDirs(json_dir)
        with tf.gfile.Open(json_path, "w") as f:
            json.dump(json_data, f, indent=4, sort_keys=True)


def get_decay_power(n_layer):
    decay_power = {
        "/word_embedding": n_layer + 1,
        "/r_w_bias": n_layer + 1,
        "/r_r_bias": n_layer + 1,
        "/r_s_bias": n_layer + 1,
        "/seg_embed": n_layer + 1,
        "/mask_emb": n_layer + 1,
        "lm_loss/": 0,
        "cls/": 0,
    }
    for layer_idx in range(n_layer):
        decay_power["/layer_%d/" % layer_idx] = n_layer - layer_idx
    return decay_power


def create_instances_from_document(
    sp, token_ids, sent_ids, max_seq_length, reuse_seq_length, batch_size, num_predict,
    mask_alpha=6, mask_beta=1, n_device=1, bi_directional=True,
):

    bsz_per_core = batch_size // n_device

    if bi_directional:
        assert batch_size % (2 * n_device) == 0, "XLNetLM requires `batch_size` evenly divided by (2 * num of CPU/GPUs)."
        fwd_data, fwd_sent_ids = batchify(token_ids, batch_size // 2, sent_ids)

        fwd_data = fwd_data.reshape(n_device, 1, bsz_per_core // 2, -1)
        fwd_sent_ids = fwd_sent_ids.reshape(n_device, 1, bsz_per_core // 2, -1)

        bwd_data = fwd_data[:, :, :, ::-1]
        bwd_sent_ids = fwd_sent_ids[:, :, :, ::-1]

        token_ids = np.concatenate([fwd_data, bwd_data], 1).reshape(batch_size, -1)
        sent_ids = np.concatenate([fwd_sent_ids, bwd_sent_ids], 1).reshape(batch_size, -1)
    else:
        token_ids, sent_ids = batchify(token_ids, batch_size, sent_ids)

    # [sep] x 2 + [cls]
    assert reuse_seq_length < max_seq_length - 3

    data_len = token_ids.shape[1]
    sep_array = np.array([SEP_ID], dtype=np.int64)
    cls_array = np.array([CLS_ID], dtype=np.int64)

    instances = []
    i = 0
    while i + max_seq_length <= data_len:

        for idx in range(batch_size):
            inp = token_ids[idx, i: i + reuse_seq_length]
            tgt = token_ids[idx, i + 1: i + reuse_seq_length + 1]

            results = _split_a_and_b(
                token_ids[idx],
                sent_ids[idx],
                begin_idx=i + reuse_seq_length,
                tot_len=max_seq_length - reuse_seq_length - 3,
                extend_target=True,
            )
            if results is None:
                break

            # unpack the results
            (a_data, b_data, label, _, a_target, b_target) = tuple(results)

            # sample ngram spans to predict
            reverse = (idx // (bsz_per_core // 2)) % 2 == 1
            if num_predict is None:
                num_predict_0 = num_predict_1 = None
            else:
                num_predict_1 = num_predict // 2
                num_predict_0 = num_predict - num_predict_1
            mask_0 = _sample_mask(
                sp, inp, mask_alpha, mask_beta,
                reverse=reverse, goal_num_predict=num_predict_0,
            )
            mask_1 = _sample_mask(
                sp, np.concatenate([a_data, sep_array, b_data, sep_array, cls_array]),
                mask_alpha, mask_beta,
                reverse=reverse, goal_num_predict=num_predict_1,
            )

            # concatenate data
            cat_data = np.concatenate([inp, a_data, sep_array, b_data, sep_array, cls_array])
            seg_id = ([0] * (reuse_seq_length + a_data.shape[0]) + [0] + [1] * b_data.shape[0] + [1] + [2])
            assert cat_data.shape[0] == max_seq_length
            assert mask_0.shape[0] == max_seq_length // 2
            assert mask_1.shape[0] == max_seq_length // 2

            # the last two CLS's are not used, just for padding purposes
            tgt = np.concatenate([tgt, a_target, b_target, cls_array, cls_array])
            assert tgt.shape[0] == max_seq_length

            is_masked = np.concatenate([mask_0, mask_1], 0)
            if num_predict is not None:
                assert np.sum(is_masked) == num_predict

            instance = {
                "input": cat_data.tolist(),
                "is_masked": is_masked.tolist(),
                "target": tgt.tolist(),
                "seg_id": seg_id,
                "label": label,
            }
            instances.append(instance)

        i += reuse_seq_length

    return instances


def _split_a_and_b(data, sent_ids, begin_idx, tot_len, extend_target=False):
    """Split two segments from `data` starting from the index `begin_idx`."""

    data_len = data.shape[0]
    if begin_idx + tot_len >= data_len:
        return None

    end_idx = begin_idx + 1
    cut_points = []
    while end_idx < data_len:
        if sent_ids[end_idx] != sent_ids[end_idx - 1]:
            if end_idx - begin_idx >= tot_len:
                break
            cut_points.append(end_idx)
        end_idx += 1

    a_begin = begin_idx
    if len(cut_points) == 0 or random.random() < 0.5:
        label = 0
        if len(cut_points) == 0:
            a_end = end_idx
        else:
            a_end = random.choice(cut_points)

        b_len = max(1, tot_len - (a_end - a_begin))
        # (zihang): `data_len - 1` to account for extend_target
        b_begin = random.randint(0, data_len - 1 - b_len)
        b_end = b_begin + b_len
        while b_begin > 0 and sent_ids[b_begin - 1] == sent_ids[b_begin]:
            b_begin -= 1
        # (zihang): `data_len - 1` to account for extend_target
        while b_end < data_len - 1 and sent_ids[b_end - 1] == sent_ids[b_end]:
            b_end += 1

        new_begin = a_end
    else:
        label = 1
        a_end = random.choice(cut_points)
        b_begin = a_end
        b_end = end_idx

        new_begin = b_end

    while a_end - a_begin + b_end - b_begin > tot_len:
        if a_end - a_begin > b_end - b_begin:
            # delete the right side only for the LM objective
            a_end -= 1
        else:
            b_end -= 1

    ret = [data[a_begin: a_end], data[b_begin: b_end], label, new_begin]

    if extend_target:
        if a_end >= data_len or b_end >= data_len:
            return None
        a_target = data[a_begin + 1: a_end + 1]
        b_target = data[b_begin: b_end + 1]
        ret.extend([a_target, b_target])

    return ret


def _sample_mask(sp, seg, mask_alpha, mask_beta, reverse=False, max_gram=5, goal_num_predict=None):
    """Sample `goal_num_predict` tokens for partial prediction.
    About `mask_beta` tokens are chosen in a context of `mask_alpha` tokens."""

    seg_len = len(seg)
    mask = np.array([False] * seg_len, dtype=np.bool)

    num_predict = 0

    ngrams = np.arange(1, max_gram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_gram + 1)
    pvals /= pvals.sum(keepdims=True)

    if reverse:
        seg = np.flip(seg, 0)

    cur_len = 0
    while cur_len < seg_len:
        if goal_num_predict is not None and num_predict >= goal_num_predict:
            break

        n = np.random.choice(ngrams, p=pvals)
        if goal_num_predict is not None:
            n = min(n, goal_num_predict - num_predict)
        ctx_size = (n * mask_alpha) // mask_beta
        l_ctx = np.random.choice(ctx_size)
        r_ctx = ctx_size - l_ctx

        # Find the start position of a complete token
        beg = cur_len + l_ctx
        while beg < seg_len and not _is_start_piece(sp.processor.IdToPiece(seg[beg].item())):
            beg += 1
        if beg >= seg_len:
            break

        # Find the end position of the n-gram (start pos of the n+1-th gram)
        end = beg + 1
        cnt_ngram = 1
        while end < seg_len:
            if _is_start_piece(sp.processor.IdToPiece(seg[beg].item())):
                cnt_ngram += 1
                if cnt_ngram > n:
                    break
            end += 1
        if end >= seg_len:
            break

        # Update
        mask[beg:end] = True
        num_predict += end - beg

        cur_len = end + r_ctx

    while goal_num_predict is not None and num_predict < goal_num_predict:
        i = np.random.randint(seg_len)
        if not mask[i]:
            mask[i] = True
            num_predict += 1

    if reverse:
        mask = np.flip(mask, 0)

    return mask


def batchify(data, bsz_per_host, sent_ids=None):
    num_step = len(data) // bsz_per_host
    data = np.array(data[:bsz_per_host * num_step])
    data = data.reshape(bsz_per_host, num_step)
    if sent_ids is not None:
        sent_ids = np.array(sent_ids[:bsz_per_host * num_step])
        sent_ids = sent_ids.reshape(bsz_per_host, num_step)

    if sent_ids is not None:
        return data, sent_ids
    return data


def _is_start_piece(piece):
    special_pieces = set(list("!\"#$%&\"()*+,-./:;?@[\\]^_`{|}~"))
    if (piece.startswith("▁") or piece.startswith("<") or piece in special_pieces):
        return True
    else:
        return False


def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
    """
    Sample a permutation of the factorization order, and create an
    attention mask accordingly.

    Args:
        inputs: int64 Tensor in shape [seq_len], input ids.
        targets: int64 Tensor in shape [seq_len], target ids.
        is_masked: bool Tensor in shape [seq_len]. True means being selected
            for partial prediction.
        perm_size: the length of longest permutation. Could be set to be reuse_len.
            Should not be larger than reuse_len or there will be data leaks.
        seq_len: int, sequence length.
    """
    batch_size = tf.shape(inputs)[0]

    # Generate permutation indices
    index = tf.range(seq_len, dtype=tf.int64)
    index = tf.reshape(index, [-1, perm_size])
    index = tf.transpose(index)
    index = tf.random_shuffle(index)
    index = tf.transpose(index)
    index = tf.reshape(index, [1, -1])
    index = tf.tile(index, [batch_size, 1])

    # `perm_mask` and `target_mask`
    # non-functional tokens
    non_func_tokens = tf.logical_not(tf.logical_or(tf.equal(inputs, SEP_ID), tf.equal(inputs, CLS_ID)))

    non_mask_tokens = tf.logical_and(tf.logical_not(is_masked), non_func_tokens)
    masked_or_func_tokens = tf.logical_not(non_mask_tokens)

    # Set the permutation indices of non-masked (& non-funcional) tokens to the
    # smallest index (-1):
    # (1) they can be seen by all other positions
    # (2) they cannot see masked positions, so there won't be information leak
    smallest_index = -tf.ones([batch_size, seq_len], dtype=tf.int64)
    rev_index = tf.where(non_mask_tokens, smallest_index, index)

    # Create `target_mask`: non-funcional and maksed tokens
    # 1: use mask as input and have loss
    # 0: use token (or [SEP], [CLS]) as input and do not have loss
    target_tokens = tf.logical_and(masked_or_func_tokens, non_func_tokens)
    target_mask = tf.cast(target_tokens, tf.float32)

    # Create `perm_mask`
    # `target_tokens` cannot see themselves
    self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)

    # 1: cannot attend if i <= j and j is not non-masked (masked_or_func_tokens)
    # 0: can attend if i > j or j is non-masked
    perm_mask = tf.logical_and(self_rev_index[:, :, None] <= rev_index[:, None, :], tf.expand_dims(masked_or_func_tokens, axis=-1))

    # new target: [next token] for LM and [curr token] (self) for PLM
    new_targets = tf.concat([inputs[:, 0: 1], targets[:, :-1]], axis=1)

    # construct inputs_k
    inputs_k = inputs

    # construct inputs_q
    inputs_q = target_mask

    return perm_mask, new_targets, target_mask, inputs_k, inputs_q


def expand_features(module, placeholders):

    inputs = placeholders["input"]
    target = placeholders["target"]
    is_masked = tf.cast(placeholders["is_masked"], tf.bool)
    batch_size = tf.shape(inputs)[0]

    non_reuse_len = module.max_seq_length - module.reuse_seq_length
    assert (module.perm_size <= module.reuse_seq_length and module.perm_size <= non_reuse_len)

    (perm_mask_0, target_0, target_mask_0, input_k_0, input_q_0) = _local_perm(
        inputs[:, :module.reuse_seq_length],
        target[:, :module.reuse_seq_length],
        is_masked[:, :module.reuse_seq_length],
        module.perm_size,
        module.reuse_seq_length,
    )

    (perm_mask_1, target_1, target_mask_1, input_k_1, input_q_1) = _local_perm(
        inputs[:, module.reuse_seq_length:],
        target[:, module.reuse_seq_length:],
        is_masked[:, module.reuse_seq_length:],
        module.perm_size,
        non_reuse_len,
    )

    perm_mask_0 = tf.concat(
        [tf.cast(perm_mask_0, dtype=tf.float32),
         tf.ones([batch_size, module.reuse_seq_length, non_reuse_len])],
        axis=2,
    )
    perm_mask_1 = tf.concat(
        [tf.zeros([batch_size, non_reuse_len, module.reuse_seq_length]),
         tf.cast(perm_mask_1, dtype=tf.float32)],
        axis=2,
    )
    perm_mask = tf.concat([perm_mask_0, perm_mask_1], axis=1)
    target = tf.concat([target_0, target_1], axis=1)
    target_mask = tf.concat([target_mask_0, target_mask_1], axis=1)
    input_k = tf.concat([input_k_0, input_k_1], axis=1)
    input_q = tf.concat([input_q_0, input_q_1], axis=1)

    if module._num_predict is not None:
        # TODO(geying): convert tensors from 1-D to 2-D

        indices = tf.range(module.max_seq_length, dtype=tf.int64)
        indices = tf.reshape(indices, [-1, module.max_seq_length])
        indices = tf.tile(indices, [batch_size, 1])
        bool_target_mask = tf.cast(target_mask, tf.bool)
        indices = tf.boolean_mask(indices, bool_target_mask)

        # extra padding due to CLS/SEP introduced after prepro
        actual_num_predict = tf.shape(indices)[1]
        pad_len = module._num_predict - actual_num_predict

        # target_mapping
        target_mapping = tf.one_hot(indices, module.max_seq_length, dtype=tf.float32)
        paddings = tf.zeros([pad_len, module.max_seq_length], dtype=target_mapping.dtype)
        target_mapping = tf.concat([target_mapping, paddings], axis=0)
        placeholders["target_mapping"] = tf.reshape(target_mapping, [-1, module._num_predict, module.max_seq_length])

        # target
        target = tf.boolean_mask(target, bool_target_mask)
        paddings = tf.zeros([pad_len], dtype=target.dtype)
        target = tf.concat([target, paddings], axis=0)
        placeholders["target"] = tf.reshape(target, [-1, module._num_predict])

        # target mask
        target_mask = tf.concat([
            tf.ones([batch_size, actual_num_predict], dtype=tf.float32),
            tf.zeros([batch_size, pad_len], dtype=tf.float32)
        ], axis=1)
        placeholders["target_mask"] = tf.reshape(target_mask, [-1, module._num_predict])
    else:
        placeholders["target"] = tf.reshape(target, [-1, module.max_seq_length])
        placeholders["target_mask"] = tf.reshape(target_mask, [-1, module.max_seq_length])

    # reshape back to fixed shape
    placeholders["perm_mask"] = tf.reshape(perm_mask, [-1, module.max_seq_length, module.max_seq_length])
    placeholders["input_k"] = tf.reshape(input_k, [-1, module.max_seq_length])
    placeholders["input_q"] = tf.reshape(input_q, [-1, module.max_seq_length])

    return placeholders
