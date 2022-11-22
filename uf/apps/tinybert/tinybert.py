""" TinyBERT, a distillation model of BERT. """

from ..bert.bert import BERTEncoder
from .._base_._base_ import BaseDecoder
from ...third import tf
from .. import util


class TinyBERTClsDistillor(BaseDecoder):
    def __init__(self,
                 student_config,
                 bert_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids=None,
                 sample_weight=None,
                 scope="bert",
                 dtype=tf.float32,
                 drop_pooler=False,
                 label_size=2,
                 trainable=True,
                 use_tilda_embedding=False,
                 **kwargs):
        super().__init__()

        def _get_logits(pooled_output, hidden_size, scope, trainable):
            with tf.variable_scope(scope):
                output_weights = tf.get_variable(
                    "output_weights",
                    shape=[label_size, hidden_size],
                    initializer=util.create_initializer(
                        bert_config.initializer_range),
                    trainable=trainable)
                output_bias = tf.get_variable(
                    "output_bias",
                    shape=[label_size],
                    initializer=tf.zeros_initializer(),
                    trainable=trainable)

                logits = tf.matmul(pooled_output,
                                   output_weights,
                                   transpose_b=True)
                logits = tf.nn.bias_add(logits, output_bias)
                return logits

        student = BERTEncoder(
            bert_config=student_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            scope="tiny/bert",
            use_tilda_embedding=use_tilda_embedding,
            drop_pooler=drop_pooler,
            trainable=True,
            **kwargs)
        student_hidden = student.get_pooled_output()
        student_logits = _get_logits(
            student_hidden,
            student_config.hidden_size, "tiny/cls/seq_relationship", trainable=trainable,
        )

        if kwargs.get("return_hidden"):
            self.tensors["hidden"] = student_hidden

        if is_training:
            teacher = BERTEncoder(
                bert_config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                scope=scope,
                use_tilda_embedding=False,
                drop_pooler=drop_pooler,
                trainable=False,
                **kwargs)
            teacher_logits = _get_logits(
                teacher.get_pooled_output(),
                bert_config.hidden_size, "cls/seq_relationship", trainable=False,
            )

            embedding_loss = self._get_embedding_loss(teacher, student, bert_config, sample_weight)
            attention_loss = self._get_attention_loss(teacher, student, bert_config, student_config, sample_weight)
            hidden_loss = self._get_hidden_loss(teacher, student, bert_config, student_config, sample_weight)
            pred_loss = self._get_pred_loss(teacher_logits, student_logits, sample_weight)
            distill_loss = embedding_loss + attention_loss + hidden_loss + pred_loss

            self.train_loss = distill_loss
            self.tensors["losses"] = tf.reshape(distill_loss, [1])

        else:
            self._infer(student_logits, label_ids, sample_weight, label_size, **kwargs)

    def _get_embedding_loss(self, teacher, student, bert_config, sample_weight):
        teacher_embedding = teacher.get_embedding_output()
        teacher_embedding = tf.stop_gradient(teacher_embedding)
        student_embedding = student.get_embedding_output()
        with tf.variable_scope("embedding_loss"):
            linear_trans = tf.layers.dense(
                student_embedding,
                bert_config.hidden_size,
                kernel_initializer=util.create_initializer(
                    bert_config.initializer_range))
            if sample_weight is not None:
                embedding_loss = tf.losses.mean_squared_error(
                    linear_trans, teacher_embedding,
                    weights=tf.reshape(sample_weight, [-1, 1, 1]))
            else:
                embedding_loss = tf.losses.mean_squared_error(
                    linear_trans, teacher_embedding)
        return embedding_loss

    def _get_attention_loss(self, teacher, student,
                            bert_config, student_config, sample_weight):
        teacher_attention_scores = teacher.get_attention_scores()
        teacher_attention_scores = [tf.stop_gradient(value)
                                    for value in teacher_attention_scores]
        student_attention_scores = student.get_attention_scores()
        num_teacher_hidden_layers = bert_config.num_hidden_layers
        num_student_hidden_layers = student_config.num_hidden_layers
        num_projections = \
            int(num_teacher_hidden_layers / num_student_hidden_layers)
        attention_losses = []
        for i in range(num_student_hidden_layers):
            if sample_weight is not None:
                attention_losses.append(tf.losses.mean_squared_error(
                    teacher_attention_scores[
                        num_projections * i + num_projections - 1],
                    student_attention_scores[i],
                    weights=tf.reshape(sample_weight, [-1, 1, 1, 1])),)
            else:
                attention_losses.append(tf.losses.mean_squared_error(
                    teacher_attention_scores[
                        num_projections * i + num_projections - 1],
                    student_attention_scores[i]),)
        attention_loss = tf.add_n(attention_losses)
        return attention_loss

    def _get_hidden_loss(self, teacher, student,
                         bert_config, student_config, sample_weight):
        teacher_hidden_layers = teacher.all_encoder_layers
        teacher_hidden_layers = [tf.stop_gradient(value)
                                 for value in teacher_hidden_layers]
        student_hidden_layers = student.all_encoder_layers
        num_teacher_hidden_layers = bert_config.num_hidden_layers
        num_student_hidden_layers = student_config.num_hidden_layers
        num_projections = int(
            num_teacher_hidden_layers / num_student_hidden_layers)
        with tf.variable_scope("hidden_loss"):
            hidden_losses = []
            for i in range(num_student_hidden_layers):
                if sample_weight is not None:
                    hidden_losses.append(tf.losses.mean_squared_error(
                        teacher_hidden_layers[
                            num_projections * i + num_projections - 1],
                        tf.layers.dense(
                            student_hidden_layers[i], bert_config.hidden_size,
                            kernel_initializer=util.create_initializer(
                                bert_config.initializer_range)),
                        weights=tf.reshape(sample_weight, [-1, 1, 1])))
                else:
                    hidden_losses.append(tf.losses.mean_squared_error(
                        teacher_hidden_layers[
                            num_projections * i + num_projections - 1],
                        tf.layers.dense(
                            student_hidden_layers[i], bert_config.hidden_size,
                            kernel_initializer=util.create_initializer(
                                bert_config.initializer_range))))
            hidden_loss = tf.add_n(hidden_losses)
        return hidden_loss

    def _get_pred_loss(self, teacher_logits, student_logits, sample_weight):
        teacher_probs = tf.nn.softmax(teacher_logits, axis=-1)
        teacher_probs = tf.stop_gradient(teacher_probs)
        student_log_probs = tf.nn.log_softmax(student_logits, axis=-1)
        pred_loss = -tf.reduce_sum(teacher_probs * student_log_probs, axis=-1)
        if sample_weight is not None:
            pred_loss *= tf.reshape(sample_weight, [-1, 1])
        pred_loss = tf.reduce_mean(pred_loss)
        return pred_loss

    def _infer(self, student_logits, label_ids, sample_weight, label_size, **kwargs):
        probs = tf.nn.softmax(student_logits, axis=-1, name="probs")
        self.tensors["probs"] = probs
        self.tensors["preds"] = tf.argmax(probs, axis=-1, name="preds")

        if label_ids is not None:
            per_example_loss = util.cross_entropy(student_logits, label_ids, label_size, **kwargs)
            if sample_weight is not None:
                per_example_loss *= sample_weight

            self.tensors["losses"] = per_example_loss


class TinyBERTBinaryClsDistillor(TinyBERTClsDistillor):

    def _get_pred_loss(self, teacher_logits, student_logits, sample_weight):
        teacher_logits = tf.stop_gradient(teacher_logits)
        pred_loss = tf.losses.mean_squared_error(teacher_logits, student_logits)
        if sample_weight is not None:
            pred_loss *= tf.reshape(sample_weight, [-1, 1])
        pred_loss = tf.reduce_mean(pred_loss)
        return pred_loss

    def _infer(self, student_logits, label_ids, sample_weight, label_size, **kwargs):
        probs = tf.nn.sigmoid(student_logits, name="probs")
        self.tensors["probs"] = probs
        self.tensors["preds"] = tf.greater(probs, 0.5, name="preds")

        if label_ids is not None:
            per_example_loss = util.sigmoid_cross_entropy(student_logits, label_ids, **kwargs)
            if sample_weight is not None:
                per_example_loss *= sample_weight

            self.tensors["losses"] = per_example_loss
