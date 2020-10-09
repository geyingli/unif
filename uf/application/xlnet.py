# coding:=utf-8
# Copyright 2020 Tencent. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
''' Applications based on XLNet. '''

import os
import random
import numpy as np

from uf.tools import tf
from .base import ClassifierModule, LMModule
from .bert import (
    BERTClassifier, BERTBinaryClassifier, BERTSeqClassifier, BERTLM)
from uf.modeling.base import CLSDecoder, BinaryCLSDecoder, SeqCLSDecoder
from uf.modeling.xlnet import XLNetEncoder, XLNet, XLNetConfig
from uf.tokenization.sentence_piece import SentencePieceTokenizer
import uf.utils as utils


SEG_ID_A   = 0
SEG_ID_B   = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4
special_symbols = {
    '<unk>'  : 0,
    '<s>'    : 1,
    '</s>'   : 2,
    '<cls>'  : 3,
    '<sep>'  : 4,
    '<pad>'  : 5,
    '<mask>' : 6,
    '<eod>'  : 7,
    '<eop>'  : 8,
}
UNK_ID = special_symbols['<unk>']
CLS_ID = special_symbols['<cls>']
SEP_ID = special_symbols['<sep>']
MASK_ID = special_symbols['<mask>']
EOD_ID = special_symbols['<eod>']



class XLNetClassifier(BERTClassifier, ClassifierModule):
    ''' Single-label classifier on XLNet. '''
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 spm_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.xlnet_config = get_xlnet_config(config_file)
        self.tokenizer = get_sentence_piece_tokenizer(spm_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(self.xlnet_config.n_layer)

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for ex_id, example in enumerate(X_target):
            try:
                segment_input_tokens.append(self._convert_x(example, tokenized))
            except Exception:
                tf.logging.warning(
                    'Wrong input format (line %d): \'%s\'. '
                    % (ex_id, example))

        # If `max_seq_length` is not mannually assigned,
        # the value will be set to the maximum length of
        # `input_ids`.
        if not self.max_seq_length:
            max_seq_length = 0
            for segments in segment_input_tokens:
                # subtract `[CLS]` and `[SEP]s`
                seq_length = sum([len(seg) + 1 for seg in segments]) + 1
                max_seq_length = max(max_seq_length, seq_length)
            self.max_seq_length = max_seq_length
            tf.logging.info('Adaptive max_seq_length: %d'
                            % self.max_seq_length)

        input_ids = []
        input_mask = []
        segment_ids = []
        for ex_id, segments in enumerate(segment_input_tokens):
            _input_ids = []
            _input_mask = []
            _segment_ids = []

            utils.truncate_segments(
                segments, self.max_seq_length - len(segments) - 1,
                truncate_method=self.truncate_method)

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_ids.extend(self.tokenizer.convert_tokens_to_ids(
                    segment) + [SEP_ID])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids.append(CLS_ID)
            _input_mask.append(1)
            _segment_ids.append(SEG_ID_CLS)

            # padding
            if len(_input_ids) < self.max_seq_length:
                delta_len = self.max_seq_length - len(_input_ids)
                _input_ids = [0] * delta_len + _input_ids
                _input_mask = [1] * delta_len + _input_mask  # it's 1, no error
                _segment_ids = [SEG_ID_PAD] * delta_len + _segment_ids

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        return input_ids, input_mask, segment_ids

    def _forward(self, is_training, split_placeholders, **kwargs):

        input_ids = tf.transpose(split_placeholders['input_ids'], [1, 0])
        input_mask = tf.transpose(split_placeholders['input_mask'], [1, 0])
        segment_ids = tf.transpose(split_placeholders['segment_ids'], [1, 0])

        encoder = XLNetEncoder(
            xlnet_config=self.xlnet_config,
            is_training=is_training,
            input_ids=input_ids,
            seg_ids=segment_ids,
            input_mask=input_mask,
            **kwargs)
        encoder_output = encoder.get_pooled_output()
        decoder = CLSDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders['label_ids'],
            label_size=self.label_size,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='cls/seq_relationship',
            name='cls',
            trainable=True,
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)



class XLNetBinaryClassifier(BERTBinaryClassifier, ClassifierModule):
    ''' Multi-label classifier on XLNet. '''
    _INFER_ATTRIBUTES = BERTBinaryClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 spm_file,
                 max_seq_length=128,
                 label_size=None,
                 label_weight=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.label_weight = label_weight
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.xlnet_config = get_xlnet_config(config_file)
        self.tokenizer = get_sentence_piece_tokenizer(spm_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(self.xlnet_config.n_layer)

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for ex_id, example in enumerate(X_target):
            try:
                segment_input_tokens.append(self._convert_x(example, tokenized))
            except Exception:
                tf.logging.warning(
                    'Wrong input format (line %d): \'%s\'. '
                    % (ex_id, example))

        # If `max_seq_length` is not mannually assigned,
        # the value will be set to the maximum length of
        # `input_ids`.
        if not self.max_seq_length:
            max_seq_length = 0
            for segments in segment_input_tokens:
                # subtract `[CLS]` and `[SEP]s`
                seq_length = sum([len(seg) + 1 for seg in segments]) + 1
                max_seq_length = max(max_seq_length, seq_length)
            self.max_seq_length = max_seq_length
            tf.logging.info('Adaptive max_seq_length: %d'
                            % self.max_seq_length)

        input_ids = []
        input_mask = []
        segment_ids = []
        for ex_id, segments in enumerate(segment_input_tokens):
            _input_ids = []
            _input_mask = []
            _segment_ids = []

            utils.truncate_segments(
                segments, self.max_seq_length - len(segments) - 1,
                truncate_method=self.truncate_method)

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_ids.extend(self.tokenizer.convert_tokens_to_ids(
                    segment) + [SEP_ID])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids.append(CLS_ID)
            _input_mask.append(1)
            _segment_ids.append(SEG_ID_CLS)

            # padding
            if len(_input_ids) < self.max_seq_length:
                delta_len = self.max_seq_length - len(_input_ids)
                _input_ids = [0] * delta_len + _input_ids
                _input_mask = [1] * delta_len + _input_mask  # it's 1, no error
                _segment_ids = [SEG_ID_PAD] * delta_len + _segment_ids

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        return input_ids, input_mask, segment_ids

    def _forward(self, is_training, split_placeholders, **kwargs):

        input_ids = tf.transpose(split_placeholders['input_ids'], [1, 0])
        input_mask = tf.transpose(split_placeholders['input_mask'], [1, 0])
        segment_ids = tf.transpose(split_placeholders['segment_ids'], [1, 0])

        encoder = XLNetEncoder(
            xlnet_config=self.xlnet_config,
            is_training=is_training,
            input_ids=input_ids,
            seg_ids=segment_ids,
            input_mask=input_mask,
            trainable=True,
            **kwargs)
        encoder_output = encoder.get_pooled_output()
        decoder = BinaryCLSDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders['label_ids'],
            label_size=self.label_size,
            sample_weight=split_placeholders.get('sample_weight'),
            label_weight=self.label_weight,
            scope='cls/seq_relationship',
            name='cls',
            trainable=True,
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)



class XLNetSeqClassifier(BERTSeqClassifier, ClassifierModule):
    ''' Sequence labeling classifier on XLNet. '''
    _INFER_ATTRIBUTES = BERTSeqClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 spm_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.xlnet_config = get_xlnet_config(config_file)
        self.tokenizer = get_sentence_piece_tokenizer(spm_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(self.xlnet_config.n_layer)

    def _convert_X(self, X_target, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for ex_id, example in enumerate(X_target):
            try:
                segment_input_tokens.append(
                    self._convert_x(example, tokenized))
            except Exception:
                tf.logging.warning(
                    'Wrong input format (line %d): \'%s\'. '
                    % (ex_id, example))

        # If `max_seq_length` is not mannually assigned,
        # the value will be set to the maximum length of
        # `input_ids`.
        if not self.max_seq_length:
            max_seq_length = 0
            for segments in segment_input_tokens:
                # subtract `[CLS]` and `[SEP]s`
                seq_length = sum([len(seg) + 1 for seg in segments]) + 1
                max_seq_length = max(max_seq_length, seq_length)
            self.max_seq_length = max_seq_length
            tf.logging.info('Adaptive max_seq_length: %d'
                            % self.max_seq_length)

        input_ids = []
        input_mask = []
        segment_ids = []
        for ex_id, segments in enumerate(segment_input_tokens):
            _input_ids = []
            _input_mask = []
            _segment_ids = []

            utils.truncate_segments(
                segments, self.max_seq_length - len(segments) - 1,
                truncate_method=self.truncate_method)

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_ids.extend(self.tokenizer.convert_tokens_to_ids(
                    segment) + [SEP_ID])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            _input_ids.append(CLS_ID)
            _input_mask.append(1)
            _segment_ids.append(SEG_ID_CLS)

            # padding
            if len(_input_ids) < self.max_seq_length:
                delta_len = self.max_seq_length - len(_input_ids)
                _input_ids = [0] * delta_len + _input_ids
                _input_mask = [1] * delta_len + _input_mask  # it's 1, no error
                _segment_ids = [SEG_ID_PAD] * delta_len + _segment_ids

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)

        return input_ids, input_mask, segment_ids

    def _forward(self, is_training, split_placeholders, **kwargs):

        input_ids = tf.transpose(split_placeholders['input_ids'], [1, 0])
        input_mask = tf.transpose(split_placeholders['input_mask'], [1, 0])
        segment_ids = tf.transpose(split_placeholders['segment_ids'], [1, 0])

        encoder = XLNetEncoder(
            xlnet_config=self.xlnet_config,
            is_training=is_training,
            input_ids=input_ids,
            seg_ids=segment_ids,
            input_mask=input_mask,
            trainable=True,
            **kwargs)
        encoder_output = encoder.get_sequence_output()
        decoder = SeqCLSDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            input_mask=split_placeholders['input_mask'],
            label_ids=split_placeholders['label_ids'],
            label_size=self.label_size,
            sample_weight=split_placeholders.get('sample_weight'),
            scope='cls/sequence',
            name='cls',
            trainable=True,
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)



class XLNetLM(BERTLM, LMModule):
    ''' Language modeling on XLNet. '''
    _INFER_ATTRIBUTES = BERTLM._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 spm_file,
                 max_seq_length=128,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 reuse_seq_length=None,
                 perm_size=None,
                 mask_alpha=6,
                 mask_beta=1,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        raise Exception(
            'We are faced with some problems in XLNetLM. '
            'It will soon be fixed in the future.')

        super(LMModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 64
        self.max_seq_length = max_seq_length
        self.reuse_seq_length = \
            reuse_seq_length if reuse_seq_length else max_seq_length // 2
        self.perm_size = \
            perm_size if perm_size else max_seq_length // 2
        self._mems = None
        self._mask_alpha = mask_alpha
        self._mask_beta = mask_beta
        self._num_predict = None
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.xlnet_config = get_xlnet_config(config_file)
        self.tokenizer = get_sentence_piece_tokenizer(spm_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(self.xlnet_config.n_layer)

    def predict(self, *args, **kwargs):
        raise AttributeError(
            '`predict` method is temporarily not supported for XLNetLM. '
            'We will try to implement in the future.')

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is None, (
                'XLNet uses permutation language modeling, which is '
                'unsupervised. `y` should be None.')

        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized
            (inputs, targets, seg_ids, labels, is_masked) = \
                self._convert_X(
                    X_tokenized if tokenized else X,
                    is_training, tokenized=tokenized)
            data['input'] = np.array(inputs, dtype=np.int32)
            data['target'] = np.array(targets, dtype=np.int32)
            data['seg_id'] = np.array(seg_ids, dtype=np.int32)
            data['label'] = np.array(labels, dtype=np.int32)
            data['is_masked'] = np.array(is_masked, dtype=np.int32)
            n_inputs = len(inputs)

            if n_inputs and n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight (fit)
        if is_training:
            sample_weight = self._convert_sample_weight(
                sample_weight, n_inputs)
            data['sample_weight'] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, is_training, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for ex_id, example in enumerate(X_target):
            try:
                segment_input_tokens.append(
                    self._convert_x(example, tokenized))
            except Exception:
                tf.logging.warning(
                    'Wrong input format (line %d): \'%s\'. '
                    % (ex_id, example))

        # If `max_seq_length` is not mannually assigned,
        # the value will be set to the maximum length of
        # `input_ids`.
        if not self.max_seq_length:
            max_seq_length = 0
            for segments in segment_input_tokens:
                # subtract `[CLS]` and `[SEP]s`
                seq_length = sum([len(seg) + 1 for seg in segments]) + 1
                max_seq_length = max(max_seq_length, seq_length)
            self.max_seq_length = max_seq_length
            tf.logging.info('Adaptive max_seq_length: %d'
                            % self.max_seq_length)

        # assign sentence id
        token_ids = []
        sent_ids = []
        sent_id = True
        for segments in segment_input_tokens:
            for segment in segments:
                cur_sent = self.tokenizer.convert_tokens_to_ids(segment)

                token_ids.extend(cur_sent)
                sent_ids.extend([sent_id] * len(cur_sent))
                sent_id = not sent_id
            token_ids.extend([EOD_ID])
            sent_ids.extend([sent_id])
            sent_id = not sent_id

        # random sampling of next sentence
        instances = create_instances_from_document(
            sp=self.tokenizer,
            token_ids=token_ids,
            sent_ids=sent_ids,
            max_seq_length=self.max_seq_length,
            reuse_seq_length=self.reuse_seq_length,
            batch_size=max(2, len(self._gpu_ids)),
            num_predict=self._num_predict,
            mask_alpha=self._mask_alpha,
            mask_beta=self._mask_beta,
            n_device=max(1, len(self._gpu_ids)))

        # aggregate
        inputs = []
        targets = []
        seg_ids = []
        labels = []
        is_masked = []
        for instance in instances:
            inputs.append(instance['input'])
            targets.append(instance['target'])
            seg_ids.append(instance['seg_id'])
            labels.append(instance['label'])
            is_masked.append(instance['is_masked'])

        return (inputs, targets, seg_ids, labels, is_masked)

    def _set_placeholders(self, target, on_export=False):
        self.placeholders = {
            'input': utils.get_placeholder(
                target, 'input',
                [None, self.max_seq_length], tf.int32),
            'target': utils.get_placeholder(
                target, 'target',
                [None, self.max_seq_length], tf.int32),
            'seg_id': utils.get_placeholder(
                target, 'seg_id',
                [None, self.max_seq_length], tf.int32),
            'label': utils.get_placeholder(
                target, 'label',
                [None], tf.int32),
            'is_masked': utils.get_placeholder(
                target, 'is_masked',
                [None, self.max_seq_length], tf.int32),
        }
        if not on_export:
            self.placeholders['sample_weight'] = \
                utils.get_placeholder(
                    target, 'sample_weight',
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        split_placeholders = _expand_features(
            self, split_placeholders)

        input_k = tf.transpose(split_placeholders['input_k'], [1, 0])
        input_q = tf.transpose(split_placeholders['input_q'], [1, 0])
        seg_id = tf.transpose(split_placeholders['seg_id'], [1, 0])
        perm_mask = tf.transpose(split_placeholders['perm_mask'], [1, 2, 0])
        target = split_placeholders['target']
        target_mask = split_placeholders['target_mask']

        target_mapping = None
        if 'target_mapping' in split_placeholders:
            target_mapping = tf.transpose(
                split_placeholders['target_mapping'], [1, 2, 0])

        model = XLNet(
            xlnet_config=self.xlnet_config,
            is_training=is_training,
            input_ids=input_k,
            seg_ids=seg_id,
            input_mask=None,
            mems=self._mems,
            perm_mask=perm_mask,
            target=target,
            target_mask=target_mask,
            target_mapping=target_mapping,
            inp_q=input_q,
            sample_weight=split_placeholders.get('sample_weight'),
            **kwargs)
        (total_loss, losses, probs, preds) = model.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op,
               self._preds['PLM'], self._preds['PLM_mask'],
               self._losses['PLM']]
        if as_feature:
            ops.extend(
                [self.placeholders['target']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_plm_labels = output_arrays[-1]
        else:
            batch_plm_labels = feed_dict[self.placeholders['target']]

        # PLM accuracy
        batch_plm_preds = output_arrays[1]
        batch_plm_mask = output_arrays[2]
        plm_accuracy = (
            np.sum((batch_plm_preds == batch_plm_labels) * batch_plm_mask) /
            batch_plm_mask.sum())
        print(batch_plm_preds[0], batch_plm_preds.shape)
        print(batch_plm_labels[0], batch_plm_labels.shape)
        print(batch_plm_mask[0], batch_plm_mask.shape)

        # PLM loss
        batch_plm_losses = output_arrays[3]
        plm_loss = np.mean(batch_plm_losses)

        info = ''
        info += ', PLM accuracy %.4f' % plm_accuracy
        info += ', PLM loss %.6f' % plm_loss

        return info

    def _get_predict_ops(self):
        return [self._preds['PLM']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # PLM preds
        plm_preds = utils.transform(output_arrays[0], n_inputs).tolist()
        plm_preds = [
            self.tokenizer.convert_ids_to_tokens(line) for line in plm_preds]

        outputs = {}
        outputs['plm_preds'] = plm_preds

        return outputs


def get_xlnet_config(config_file=None):
    if not os.path.exists(config_file):
        raise ValueError(
            'Can\'t find config_file \'%s\'. '
            'Please pass the correct path of configuration file, '
            'e.g.`xlnet_config.json`. An example can be downloaded from '
            'https://github.com/zihangdai/xlnet.' % config_file)
    return XLNetConfig(json_path=config_file)


def get_sentence_piece_tokenizer(spm_file, do_lower_case=True):
    if not os.path.exists(spm_file):
        raise ValueError(
            'Can\'t find vocab_file \'%s\'. '
            'Please pass the correct path of sentence-piece model file, '
            'e.g.`spiece.model`. An example can be downloaded from '
            'https://github.com/zihangdai/xlnet.' % spm_file)
    return SentencePieceTokenizer(spm_file, do_lower_case=do_lower_case)


def get_key_to_depths(n_layer):
    key_to_depths = {
        '/word_embedding': n_layer + 1,
        '/r_w_bias': n_layer + 1,
        '/r_r_bias': n_layer + 1,
        '/r_s_bias': n_layer + 1,
        '/seg_embed': n_layer + 1,
        '/mask_emb': n_layer + 1,
        'lm_loss/': 0,
        'cls/': 0}
    for layer_idx in range(n_layer):
        key_to_depths['/layer_%d/' % layer_idx] = n_layer - layer_idx
    return key_to_depths


def create_instances_from_document(sp, token_ids, sent_ids,
                                   max_seq_length, reuse_seq_length,
                                   batch_size, num_predict,
                                   mask_alpha=6, mask_beta=1,
                                   n_device=1,
                                   bi_directional=True):

    bsz_per_core = batch_size // n_device

    if bi_directional:
        assert batch_size % (2 * n_device) == 0, (
            'XLNetLM requires `batch_size` evenly divided by '
            '(2 * num of CPU/GPUs).')
        fwd_data, fwd_sent_ids = batchify(token_ids, batch_size // 2, sent_ids)

        fwd_data = fwd_data.reshape(n_device, 1, bsz_per_core // 2, -1)
        fwd_sent_ids = fwd_sent_ids.reshape(n_device, 1, bsz_per_core // 2, -1)

        bwd_data = fwd_data[:, :, :, ::-1]
        bwd_sent_ids = fwd_sent_ids[:, :, :, ::-1]

        token_ids = np.concatenate(
            [fwd_data, bwd_data], 1).reshape(batch_size, -1)
        sent_ids = np.concatenate(
            [fwd_sent_ids, bwd_sent_ids], 1).reshape(batch_size, -1)
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
                extend_target=True)
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
                reverse=reverse, goal_num_predict=num_predict_0)
            mask_1 = _sample_mask(
                sp, np.concatenate(
                    [a_data, sep_array, b_data, sep_array, cls_array]),
                mask_alpha, mask_beta,
                reverse=reverse, goal_num_predict=num_predict_1)

            # concatenate data
            cat_data = np.concatenate([inp, a_data, sep_array, b_data,
                                       sep_array, cls_array])
            seg_id = ([0] * (reuse_seq_length + a_data.shape[0]) + [0] +
                      [1] * b_data.shape[0] + [1] + [2])
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
                'input': cat_data.tolist(),
                'is_masked': is_masked.tolist(),
                'target': tgt.tolist(),
                'seg_id': seg_id,
                'label': label,
            }
            instances.append(instance)

        i += reuse_seq_length

    return instances


def _split_a_and_b(data, sent_ids, begin_idx, tot_len, extend_target=False):
  '''Split two segments from `data` starting from the index `begin_idx`.'''

  data_len = data.shape[0]
  if begin_idx + tot_len >= data_len:
    return None

  end_idx = begin_idx + 1
  cut_points = []
  while end_idx < data_len:
    if sent_ids[end_idx] != sent_ids[end_idx - 1]:
      if end_idx - begin_idx >= tot_len: break
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


def _sample_mask(sp, seg, mask_alpha, mask_beta,
                 reverse=False, max_gram=5, goal_num_predict=None):
  '''Sample `goal_num_predict` tokens for partial prediction.
  About `mask_beta` tokens are chosen in a context of `mask_alpha` tokens.'''

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
    if goal_num_predict is not None and num_predict >= goal_num_predict: break

    n = np.random.choice(ngrams, p=pvals)
    if goal_num_predict is not None:
      n = min(n, goal_num_predict - num_predict)
    ctx_size = (n * mask_alpha) // mask_beta
    l_ctx = np.random.choice(ctx_size)
    r_ctx = ctx_size - l_ctx

    # Find the start position of a complete token
    beg = cur_len + l_ctx
    while beg < seg_len and \
        not _is_start_piece(sp.processor.IdToPiece(seg[beg].item())):
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
  special_pieces = set(list('!"#$%&\"()*+,-./:;?@[\\]^_`{|}~'))
  if (piece.startswith("▁") or piece.startswith("<")
      or piece in special_pieces):
    return True
  else:
    return False


def _local_perm(inputs, targets, is_masked, perm_size, seq_len):
  '''
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
  '''
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
  non_func_tokens = tf.logical_not(tf.logical_or(
      tf.equal(inputs, SEP_ID),
      tf.equal(inputs, CLS_ID)))

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
  perm_mask = tf.logical_and(
      self_rev_index[:, :, None] <= rev_index[:, None, :],
      tf.expand_dims(masked_or_func_tokens, axis=-1))

  # new target: [next token] for LM and [curr token] (self) for PLM
  new_targets = tf.concat([inputs[:, 0: 1], targets[:, :-1]], axis=1)

  # construct inputs_k
  inputs_k = inputs

  # construct inputs_q
  inputs_q = target_mask

  return perm_mask, new_targets, target_mask, inputs_k, inputs_q


def _expand_features(module, split_placeholders):

    inputs = split_placeholders['input']
    target = split_placeholders['target']
    is_masked = tf.cast(split_placeholders['is_masked'], tf.bool)
    batch_size = tf.shape(inputs)[0]

    non_reuse_len = module.max_seq_length - module.reuse_seq_length
    assert (module.perm_size <= module.reuse_seq_length and
            module.perm_size <= non_reuse_len)

    (perm_mask_0, target_0, target_mask_0, input_k_0, input_q_0) = \
        _local_perm(
            inputs[:, :module.reuse_seq_length],
            target[:, :module.reuse_seq_length],
            is_masked[:, :module.reuse_seq_length],
            module.perm_size,
            module.reuse_seq_length)

    (perm_mask_1, target_1, target_mask_1, input_k_1, input_q_1) = \
        _local_perm(
            inputs[:, module.reuse_seq_length:],
            target[:, module.reuse_seq_length:],
            is_masked[:, module.reuse_seq_length:],
            module.perm_size,
            non_reuse_len)

    perm_mask_0 = tf.concat(
        [tf.cast(perm_mask_0, dtype=tf.float32),
         tf.ones([batch_size, module.reuse_seq_length, non_reuse_len])],
        axis=2)
    perm_mask_1 = tf.concat(
        [tf.zeros([batch_size, non_reuse_len, module.reuse_seq_length]),
         tf.cast(perm_mask_1, dtype=tf.float32)],
        axis=2)
    perm_mask = tf.concat([perm_mask_0, perm_mask_1], axis=1)
    target = tf.concat([target_0, target_1], axis=1)
    target_mask = tf.concat([target_mask_0, target_mask_1], axis=1)
    input_k = tf.concat([input_k_0, input_k_1], axis=1)
    input_q = tf.concat([input_q_0, input_q_1], axis=1)

    if module._num_predict is not None:
        #TODO(geying): convert tensors from 1-D to 2-D

        indices = tf.range(module.max_seq_length, dtype=tf.int64)
        indices = tf.reshape(indices, [-1, module.max_seq_length])
        indices = tf.tile(indices, [batch_size, 1])
        bool_target_mask = tf.cast(target_mask, tf.bool)
        indices = tf.boolean_mask(indices, bool_target_mask)

        ##### extra padding due to CLS/SEP introduced after prepro
        actual_num_predict = tf.shape(indices)[1]
        pad_len = module._num_predict - actual_num_predict

        ##### target_mapping
        target_mapping = tf.one_hot(
            indices, module.max_seq_length, dtype=tf.float32)
        paddings = tf.zeros([pad_len, module.max_seq_length],
                            dtype=target_mapping.dtype)
        target_mapping = tf.concat([target_mapping, paddings], axis=0)
        split_placeholders['target_mapping'] = tf.reshape(
            target_mapping, [-1, module._num_predict, module.max_seq_length])

        ##### target
        target = tf.boolean_mask(target, bool_target_mask)
        paddings = tf.zeros([pad_len], dtype=target.dtype)
        target = tf.concat([target, paddings], axis=0)
        split_placeholders['target'] = tf.reshape(
            target, [-1, module._num_predict])

        ##### target mask
        target_mask = tf.concat(
            [tf.ones([batch_size, actual_num_predict], dtype=tf.float32),
             tf.zeros([batch_size, pad_len], dtype=tf.float32)],
            axis=1)
        split_placeholders['target_mask'] = tf.reshape(
            target_mask, [-1, module._num_predict])
    else:
        split_placeholders['target'] = tf.reshape(
            target, [-1, module.max_seq_length])
        split_placeholders['target_mask'] = tf.reshape(
            target_mask, [-1, module.max_seq_length])

    # reshape back to fixed shape
    split_placeholders['perm_mask'] = tf.reshape(
        perm_mask, [-1, module.max_seq_length, module.max_seq_length])
    split_placeholders['input_k'] = tf.reshape(
        input_k, [-1, module.max_seq_length])
    split_placeholders['input_q'] = tf.reshape(
        input_q, [-1, module.max_seq_length])

    return split_placeholders