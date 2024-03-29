import numpy as np

from .xlnet import XLNet, XLNetConfig, create_instances_from_document, expand_features, get_decay_power, EOD_ID
from .._base_._base_classifier import LMModule
from ..bert.bert_lm import BERTLM
from ...token import SentencePieceTokenizer
from ...third import tf
from ... import com


class XLNetLM(LMModule):
    """ Language modeling on XLNet. """

    def __init__(
        self,
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
        truncate_method="LIFO",
    ):
        raise Exception("We are faced with some problems in XLNetLM. It may be fixed in the future.")

        self.__init_args__ = locals()
        super(LMModule, self).__init__(init_checkpoint, output_dir, gpu_ids)

        self.batch_size = 64
        self.max_seq_length = max_seq_length
        self.reuse_seq_length = reuse_seq_length if reuse_seq_length else max_seq_length // 2
        self.perm_size = perm_size if perm_size else max_seq_length // 2
        self.truncate_method = truncate_method
        self._mems = None
        self._mask_alpha = mask_alpha
        self._mask_beta = mask_beta
        self._num_predict = None

        self.xlnet_config = XLNetConfig(json_path=config_file)
        self.tokenizer = SentencePieceTokenizer(spm_file, do_lower_case)
        self.decay_power = get_decay_power(self.xlnet_config.n_layer)

    def predict(self, *args, **kwargs):
        raise AttributeError("`predict` method is temporarily not supported for XLNetLM. We will try to implement in the future.")

    def convert(self, X=None, y=None, sample_weight=None, X_tokenized=None, is_training=False, is_parallel=False):
        self._assert_legal(X, y, sample_weight, X_tokenized)

        if is_training:
            assert y is None, "Training of %s is unsupervised. `y` should be None." % self.__class__.__name__

        n_inputs = None
        data = {}

        # convert X
        if X is not None or X_tokenized is not None:
            tokenized = False if X is not None else X_tokenized
            inputs, targets, seg_ids, labels, is_masked = self._convert_X(X_tokenized if tokenized else X, is_training, tokenized=tokenized)
            data["input"] = np.array(inputs, dtype=np.int32)
            data["target"] = np.array(targets, dtype=np.int32)
            data["seg_id"] = np.array(seg_ids, dtype=np.int32)
            data["label"] = np.array(labels, dtype=np.int32)
            data["is_masked"] = np.array(is_masked, dtype=np.int32)
            n_inputs = len(inputs)

            if n_inputs and n_inputs < self.batch_size:
                self.batch_size = max(n_inputs, len(self._gpu_ids))

        # convert sample_weight
        if is_training or y is not None:
            sample_weight = self._convert_sample_weight(sample_weight, n_inputs)
            data["sample_weight"] = np.array(sample_weight, dtype=np.float32)

        return data

    def _convert_X(self, X_target, is_training, tokenized):

        # tokenize input texts
        segment_input_tokens = []
        for idx, sample in enumerate(X_target):
            try:
                segment_input_tokens.append(self._convert_x(sample, tokenized))
            except Exception as e:
                raise ValueError("Wrong input format (%s): %s." % (sample, e))

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
            n_device=max(1, len(self._gpu_ids)),
        )

        # aggregate
        inputs = []
        targets = []
        seg_ids = []
        labels = []
        is_masked = []
        for instance in instances:
            inputs.append(instance["input"])
            targets.append(instance["target"])
            seg_ids.append(instance["seg_id"])
            labels.append(instance["label"])
            is_masked.append(instance["is_masked"])

        return (inputs, targets, seg_ids, labels, is_masked)

    def _set_placeholders(self, **kwargs):
        self.placeholders = {
            "input": tf.placeholder(tf.int32, [None, self.max_seq_length], "input"),
            "target": tf.placeholder(tf.int32, [None, self.max_seq_length], "target"),
            "seg_id": tf.placeholder(tf.int32, [None, self.max_seq_length], "seg_id"),
            "label": tf.placeholder(tf.int32, [None], "label"),
            "is_masked": tf.placeholder(tf.int32, [None, self.max_seq_length], "is_masked"),
            "sample_weight": tf.placeholder(tf.float32, [None], "sample_weight"),
        }

    def _forward(self, is_training, placeholders, **kwargs):

        placeholders = expand_features(self, placeholders)

        input_k = tf.transpose(placeholders["input_k"], [1, 0])
        input_q = tf.transpose(placeholders["input_q"], [1, 0])
        seg_id = tf.transpose(placeholders["seg_id"], [1, 0])
        perm_mask = tf.transpose(placeholders["perm_mask"], [1, 2, 0])
        target = placeholders["target"]
        target_mask = placeholders["target_mask"]

        target_mapping = None
        if "target_mapping" in placeholders:
            target_mapping = tf.transpose(placeholders["target_mapping"], [1, 2, 0])

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
            sample_weight=placeholders.get("sample_weight"),
            **kwargs,
        )
        train_loss, tensors = model.get_forward_outputs()
        return train_loss, tensors

    def _get_fit_ops(self, from_tfrecords=False):
        ops = [self.tensors["preds"], self.tensors["mask"], self.tensors["losses"]]
        if from_tfrecords:
            ops.extend([self.placeholders["target"]])
        return ops

    def _get_fit_info(self, output_arrays, feed_dict, from_tfrecords=False):

        if from_tfrecords:
            batch_plm_labels = output_arrays[-1]
        else:
            batch_plm_labels = feed_dict[self.placeholders["target"]]

        # PLM accuracy
        batch_plm_preds = output_arrays[0]
        batch_plm_mask = output_arrays[1]
        plm_accuracy = np.sum((batch_plm_preds == batch_plm_labels) * batch_plm_mask) / batch_plm_mask.sum()
        print(batch_plm_preds[0], batch_plm_preds.shape)
        print(batch_plm_labels[0], batch_plm_labels.shape)
        print(batch_plm_mask[0], batch_plm_mask.shape)

        # PLM loss
        batch_plm_losses = output_arrays[2]
        plm_loss = np.mean(batch_plm_losses)

        info = ""
        info += ", PLM accuracy %.4f" % plm_accuracy
        info += ", PLM loss %.6f" % plm_loss

        return info

    def _get_predict_ops(self):
        return [self.tensors["preds"]]

    def _get_predict_outputs(self, output_arrays, n_inputs):

        # PLM preds
        plm_preds = com.transform(output_arrays[0], n_inputs).tolist()
        plm_preds = [self.tokenizer.convert_ids_to_tokens(line) for line in plm_preds]

        outputs = {}
        outputs["plm_preds"] = plm_preds

        return outputs
