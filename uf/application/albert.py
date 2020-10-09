# coding:=utf-8
# Copyright 2020 Tencent. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
''' Applications based on ALBERT. '''

import os
import copy
import random
import collections
import numpy as np

from uf.tools import tf
from .base import ClassifierModule, MRCModule, LMModule
from uf.modeling.albert import ALBERTEncoder, ALBERTDecoder, ALBERTConfig
from .bert import (BERTClassifier, BERTBinaryClassifier, BERTSeqClassifier,
                   BERTMRC, BERTLM)
from uf.modeling.base import (CLSDecoder, BinaryCLSDecoder, SeqCLSDecoder,
                              MRCDecoder)
from uf.tokenization.word_piece import WordPieceTokenizer
import uf.utils as utils



class ALBERTClassifier(BERTClassifier, ClassifierModule):
    ''' Single-label classifier on ALBERT. '''
    _INFER_ATTRIBUTES = BERTClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 drop_pooler=False,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self._drop_pooler = drop_pooler
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.albert_config = get_albert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.albert_config.num_hidden_layers)

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = ALBERTEncoder(
            config=self.albert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            token_type_ids=split_placeholders['segment_ids'],
            scope='bert',
            drop_pooler=self._drop_pooler,
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
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)



class ALBERTBinaryClassifier(BERTBinaryClassifier, ClassifierModule):
    ''' Multi-label classifier on ALBERT. '''
    _INFER_ATTRIBUTES = BERTBinaryClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 label_size=None,
                 label_weight=None,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 drop_pooler=False,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(ClassifierModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = label_size
        self.label_weight = label_weight
        self._drop_pooler = drop_pooler
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.albert_config = get_albert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.albert_config.num_hidden_layers)

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = ALBERTEncoder(
            config=self.albert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            token_type_ids=split_placeholders['segment_ids'],
            scope='bert',
            drop_pooler=self._drop_pooler,
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
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)



class ALBERTSeqClassifier(BERTSeqClassifier, ClassifierModule):
    ''' Sequence labeling classifier on ALBERT. '''
    _INFER_ATTRIBUTES = BERTSeqClassifier._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
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

        self.albert_config = get_albert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.albert_config.num_hidden_layers)

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = ALBERTEncoder(
            config=self.albert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            token_type_ids=split_placeholders['segment_ids'],
            scope='bert',
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
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)



class ALBERTMRC(BERTMRC, MRCModule):
    ''' Machine reading comprehension on ALBERT. '''
    _INFER_ATTRIBUTES = BERTMRC._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=256,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 do_lower_case=True,
                 truncate_method='longer-FO'):
        super(MRCModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.albert_config = get_albert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.albert_config.num_hidden_layers)

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = ALBERTEncoder(
            config=self.albert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            token_type_ids=split_placeholders['segment_ids'],
            scope='bert',
            **kwargs)
        encoder_output = encoder.get_sequence_output()
        decoder = MRCDecoder(
            is_training=is_training,
            input_tensor=encoder_output,
            label_ids=split_placeholders['label_ids'],
            sample_weight=split_placeholders.get('sample_weight'),
            scope='mrc',
            name='mrc',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)



class ALBERTLM(BERTLM, LMModule):
    ''' Language modeling on ALBERT. '''
    _INFER_ATTRIBUTES = BERTLM._INFER_ATTRIBUTES

    def __init__(self,
                 config_file,
                 vocab_file,
                 max_seq_length=128,
                 init_checkpoint=None,
                 output_dir=None,
                 gpu_ids=None,
                 drop_pooler=False,
                 do_sample_sentence=True,
                 max_predictions_per_seq=20,
                 dupe_factor=1,
                 masked_lm_prob=0.15,
                 short_seq_prob=0.1,
                 n_gram=3,
                 favor_shorter_ngram=True,
                 do_permutation=False,
                 do_whole_word_mask=True,
                 do_lower_case=True,
                 truncate_method='LIFO'):
        super(LMModule, self).__init__(
            init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 0
        self.max_seq_length = max_seq_length
        self.label_size = 2
        self._drop_pooler = drop_pooler
        self._do_sample_sentence = do_sample_sentence
        self._max_predictions_per_seq = max_predictions_per_seq
        self._dupe_factor = dupe_factor
        self._masked_lm_prob = masked_lm_prob
        self._short_seq_prob = short_seq_prob
        self._ngram = n_gram
        self._favor_shorter_ngram = favor_shorter_ngram
        self._do_permutation = do_permutation
        self._do_whole_word_mask = do_whole_word_mask
        self.truncate_method = truncate_method
        self._id_to_label = None
        self.__init_args__ = locals()

        self.albert_config = get_albert_config(config_file)
        self.tokenizer = get_word_piece_tokenizer(vocab_file, do_lower_case)
        self._key_to_depths = get_key_to_depths(
            self.albert_config.num_hidden_layers)

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None,
                is_training=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)
        n_inputs = None
        data = {}

        # convert X
        if X or X_tokenized:
            tokenized = False if X else X_tokenized

            (input_ids, input_mask, segment_ids,
             masked_lm_positions, masked_lm_ids, masked_lm_weights,
             sentence_order_labels) = self._convert_X(
                 X_tokenized if tokenized else X,
                 is_training, tokenized=tokenized)

            data['input_ids'] = np.array(input_ids, dtype=np.int32)
            data['input_mask'] = np.array(input_mask, dtype=np.int32)
            data['segment_ids'] = np.array(segment_ids, dtype=np.int32)
            data['masked_lm_positions'] = \
                np.array(masked_lm_positions, dtype=np.int32)

            if is_training:
                data['masked_lm_ids'] = \
                    np.array(masked_lm_ids, dtype=np.int32)
                data['masked_lm_weights'] = \
                    np.array(masked_lm_weights, dtype=np.float32)

            if is_training and self._do_sample_sentence:
                data['sentence_order_labels'] = \
                    np.array(sentence_order_labels, dtype=np.int32)

            n_inputs = len(input_ids)
            if n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert y
        if y:
            sentence_order_labels = self._convert_y(y, n_inputs)
            data['sentence_order_labels'] = \
                np.array(sentence_order_labels, dtype=np.int32)

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

        input_ids = []
        input_mask = []
        segment_ids = []
        masked_lm_positions = []
        masked_lm_ids = []
        masked_lm_weights = []
        sentence_order_labels = []

        # duplicate raw inputs
        if is_training and self._dupe_factor > 1:
            new_segment_input_tokens = []
            for _ in range(self._dupe_factor):
                new_segment_input_tokens.extend(
                    copy.deepcopy(segment_input_tokens))
            segment_input_tokens = new_segment_input_tokens

        # random sampling of next sentence
        if is_training and self._do_sample_sentence:
            new_segment_input_tokens = []
            for ex_id in range(len(segment_input_tokens)):
                instances = create_instances_from_document(
                    all_documents=segment_input_tokens,
                    document_index=ex_id,
                    max_seq_length=self.max_seq_length - 3,
                    masked_lm_prob=self._masked_lm_prob,
                    max_predictions_per_seq=self._max_predictions_per_seq,
                    short_seq_prob=self._short_seq_prob,
                    vocab_words=list(self.tokenizer.vocab.keys()))
                for (segments, is_random_next) in instances:
                    new_segment_input_tokens.append(segments)
                    sentence_order_labels.append(is_random_next)
            segment_input_tokens = new_segment_input_tokens

        for ex_id, segments in enumerate(segment_input_tokens):
            _input_tokens = ['[CLS]']
            _input_ids = []
            _input_mask = [1]
            _segment_ids = [0]
            _masked_lm_positions = []
            _masked_lm_ids = []
            _masked_lm_weights = []

            utils.truncate_segments(
                segments, self.max_seq_length - len(segments) - 1,
                truncate_method=self.truncate_method)

            for s_id, segment in enumerate(segments):
                _segment_id = min(s_id, 1)
                _input_tokens.extend(segment + ['[SEP]'])
                _input_mask.extend([1] * (len(segment) + 1))
                _segment_ids.extend([_segment_id] * (len(segment) + 1))

            # random sampling of masked tokens
            if is_training:
                (_input_tokens, _masked_lm_positions, _masked_lm_labels) = \
                    create_masked_lm_predictions(
                        tokens=_input_tokens,
                        masked_lm_prob=self._masked_lm_prob,
                        max_predictions_per_seq=self._max_predictions_per_seq,
                        vocab_words=list(self.tokenizer.vocab.keys()),
                        ngram=self._ngram,
                        favor_shorter_ngram=self._favor_shorter_ngram,
                        do_permutation=self._do_permutation,
                        do_whole_word_mask=self._do_whole_word_mask)
                _masked_lm_ids = \
                    self.tokenizer.convert_tokens_to_ids(_masked_lm_labels)
                _masked_lm_weights = [1.0] * len(_masked_lm_positions)

                # padding
                for _ in range(self._max_predictions_per_seq *
                               (1 + self._do_permutation) -
                               len(_masked_lm_positions)):
                    _masked_lm_positions.append(0)
                    _masked_lm_ids.append(0)
                    _masked_lm_weights.append(0.0)
            else:
                # `masked_lm_positions` is required for both training
                # and inference of BERT language modeling.
                for i in range(len(_input_tokens)):
                    if _input_tokens[i] == '[MASK]':
                        _masked_lm_positions.append(i)

                # padding
                for _ in range(self._max_predictions_per_seq -
                               len(_masked_lm_positions)):
                    _masked_lm_positions.append(0)

            _input_ids = self.tokenizer.convert_tokens_to_ids(_input_tokens)

            # padding
            for _ in range(self.max_seq_length - len(_input_ids)):
                _input_ids.append(0)
                _input_mask.append(0)
                _segment_ids.append(0)

            input_ids.append(_input_ids)
            input_mask.append(_input_mask)
            segment_ids.append(_segment_ids)
            masked_lm_positions.append(_masked_lm_positions)
            masked_lm_ids.append(_masked_lm_ids)
            masked_lm_weights.append(_masked_lm_weights)

        return (input_ids, input_mask, segment_ids,
                masked_lm_positions, masked_lm_ids, masked_lm_weights,
                sentence_order_labels)

    def _set_placeholders(self, target, on_export=False):
        self.placeholders = {
            'input_ids': utils.get_placeholder(
                target, 'input_ids',
                [None, self.max_seq_length], tf.int32),
            'input_mask': utils.get_placeholder(
                target, 'input_mask',
                [None, self.max_seq_length], tf.int32),
            'segment_ids': utils.get_placeholder(
                target, 'segment_ids',
                [None, self.max_seq_length], tf.int32),
            'masked_lm_positions': utils.get_placeholder(
                target, 'masked_lm_positions',
                [None, self._max_predictions_per_seq], tf.int32),
            'masked_lm_ids': utils.get_placeholder(
                target, 'masked_lm_ids',
                [None, self._max_predictions_per_seq], tf.int32),
            'masked_lm_weights': utils.get_placeholder(
                target, 'masked_lm_weights',
                [None, self._max_predictions_per_seq], tf.float32),
            'sentence_order_labels': utils.get_placeholder(
                target, 'sentence_order_labels',
                [None], tf.int32),
        }
        if not on_export:
            self.placeholders['sample_weight'] = \
                utils.get_placeholder(
                    target, 'sample_weight',
                    [None], tf.float32)

    def _forward(self, is_training, split_placeholders, **kwargs):

        encoder = ALBERTEncoder(
            config=self.albert_config,
            is_training=is_training,
            input_ids=split_placeholders['input_ids'],
            input_mask=split_placeholders['input_mask'],
            token_type_ids=split_placeholders['segment_ids'],
            scope='bert',
            drop_pooler=self._drop_pooler,
            **kwargs)
        decoder = ALBERTDecoder(
            albert_config=self.albert_config,
            is_training=is_training,
            encoder=encoder,
            masked_lm_positions=split_placeholders['masked_lm_positions'],
            masked_lm_ids=split_placeholders['masked_lm_ids'],
            masked_lm_weights=split_placeholders['masked_lm_weights'],
            sentence_order_labels=\
                split_placeholders.get('sentence_order_labels'),
            sample_weight=split_placeholders.get('sample_weight'),
            scope_lm='cls/predictions',
            scope_cls='cls/seq_relationship',
            name='SOP',
            **kwargs)
        (total_loss, losses, probs, preds) = decoder.get_forward_outputs()
        return (total_loss, losses, probs, preds)

    def _get_fit_ops(self, as_feature=False):
        ops = [self._train_op,
               self._preds['MLM'], self._preds['SOP'],
               self._losses['MLM'], self._losses['SOP']]
        if as_feature:
            ops.extend(
                [self.placeholders['masked_lm_positions'],
                 self.placeholders['masked_lm_ids'],
                 self.placeholders['sentence_order_labels']])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, as_feature=False):

        if as_feature:
            batch_mlm_positions = output_arrays[-3]
            batch_mlm_labels = output_arrays[-2]
            batch_sop_labels = output_arrays[-1]
        else:
            batch_mlm_positions = \
                feed_dict[self.placeholders['masked_lm_positions']]
            batch_mlm_labels = \
                feed_dict[self.placeholders['masked_lm_ids']]
            batch_sop_labels = \
                feed_dict[self.placeholders['sentence_order_labels']]

        # MLM accuracy
        batch_mlm_preds = output_arrays[1]
        batch_mlm_positions = np.reshape(batch_mlm_positions, [-1])
        batch_mlm_labels = np.reshape(batch_mlm_labels, [-1])
        batch_mlm_mask = (batch_mlm_positions > 0)
        mlm_accuracy = (
            np.sum((batch_mlm_preds == batch_mlm_labels) * batch_mlm_mask) /
            batch_mlm_mask.sum())

        # SOP accuracy
        batch_sop_preds = output_arrays[2]
        SOP_accuracy = np.mean(batch_sop_preds == batch_sop_labels)

        # MLM loss
        batch_mlm_losses = output_arrays[3]
        mlm_loss = np.mean(batch_mlm_losses)

        # SOP loss
        batch_SOP_losses = output_arrays[4]
        SOP_loss = np.mean(batch_SOP_losses)

        info = ''
        info += ', MLM accuracy %.4f' % mlm_accuracy
        info += ', SOP accuracy %.4f' % SOP_accuracy
        info += ', MLM loss %.6f' % mlm_loss
        info += ', SOP loss %.6f' % SOP_loss

        return info

    def _get_predict_ops(self):
        return [self._preds['MLM'], self._preds['SOP'], self._probs['SOP']]

    def _get_predict_outputs(self, batch_outputs):
        n_inputs = len(list(self.data.values())[0])
        output_arrays = list(zip(*batch_outputs))

        # MLM preds
        mlm_preds = []
        mlm_positions = self.data['masked_lm_positions']
        all_preds = utils.transform(output_arrays[0], n_inputs, reshape=True)
        for ex_id, _preds in enumerate(all_preds):
            _ids = []
            for p_id, _id in enumerate(_preds):
                if mlm_positions[ex_id][p_id] == 0:
                    break
                _ids.append(_id)
            mlm_preds.append(self.tokenizer.convert_ids_to_tokens(_ids))

        # SOP preds
        SOP_preds = utils.transform(output_arrays[1], n_inputs).tolist()

        # SOP probs
        SOP_probs = utils.transform(output_arrays[2], n_inputs)

        outputs = {}
        outputs['mlm_preds'] = mlm_preds
        outputs['sop_preds'] = SOP_preds
        outputs['sop_probs'] = SOP_probs

        return outputs


def create_instances_from_document(all_documents, document_index,
                                   max_seq_length, masked_lm_prob,
                                   max_predictions_per_seq,
                                   short_seq_prob, vocab_words,
                                   random_next_sentence=False):
    document = all_documents[document_index]

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_seq_length
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_seq_length)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into
                # the `A` (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or \
                        (random_next_sentence and random.random() < 0.5):
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for
                    # large corpora. However, just to be careful, we try to
                    # make sure that the random document is not the same as
                    # the document we're processing.
                    for _ in range(10):
                        random_document_index = random.randint(
                            0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them
                    # back" so they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                elif not random_next_sentence and random.random() < 0.5:
                    is_random_next = True
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                    tokens_a, tokens_b = tokens_b, tokens_a
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                instances.append(([tokens_a, tokens_b], is_random_next))
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens,
                                 masked_lm_prob,
                                 max_predictions_per_seq,
                                 vocab_words,
                                 ngram=3,
                                 favor_shorter_ngram=True,
                                 do_permutation=False,
                                 do_whole_word_mask=True):
    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.

    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith('##')):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    # Note(mingdachen):
    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, ngram + 1)
    pvals /= pvals.sum(keepdims=True)

    if not favor_shorter_ngram:
        pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx+n])
        ngram_indexes.append(ngram_index)

    random.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or
        # previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(ngrams[:len(cand_index_set)],
                             p=pvals[:len(cand_index_set)] /
                             pvals[:len(cand_index_set)].sum(keepdims=True))
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[
                        random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(
                index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict

    random.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or
            # previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(
                ngrams[:len(cand_index_set)],
                p=pvals[:len(cand_index_set)] /
                pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        random.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(
                index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
      masked_lm_positions.append(p.index)
      masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels)


def get_albert_config(config_file=None):
    if not os.path.exists(config_file):
        raise ValueError(
            'Can\'t find config_file \'%s\'. '
            'Please pass the correct path of configuration file, '
            'e.g.`albert_config.json`. An example can be downloaded from '
            'https://github.com/google-research/albert.' % config_file)
    return ALBERTConfig.from_json_file(config_file)


def get_word_piece_tokenizer(vocab_file, do_lower_case=True):
    if not os.path.exists(vocab_file):
        raise ValueError(
            'Can\'t find vocab_file \'%s\'. '
            'Please pass the correct path of vocabulary file, '
            'e.g.`vocab.txt`. An example can be downloaded from '
            'https://github.com/google-research/albert.' % vocab_file)
    return WordPieceTokenizer(vocab_file, do_lower_case=do_lower_case)


def get_key_to_depths(num_hidden_layers):
    key_to_depths = {
        '/embeddings': 4,
        '/embedding_hidden_mapping_in': 3,
        '/group_0': 2,
        '/pooler/': 1,
        'cls/': 0}
    return key_to_depths
