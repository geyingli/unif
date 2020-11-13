# Tutorial

### 1. Modeling

#### 1.1 New module

Modules have diverse argument choices. Taking demo as example:

```python
import uf

model = uf.BERTClassifier(config_file='demo/bert_config.json', vocab_file='demo/vocab.txt')

help(model)
# uf.application.bert.BERTClassifier(config_file, vocab_file, max_seq_length=128, label_size=None, init_checkpoint=None, output_dir=None, gpu_ids=None, drop_pooler=False, do_lower_case=True, truncate_method='LIFO')
```

- config_file: str type, required. The path of json file released with pretrained models.
- vocab_file: str type, required. The path of vocabulary released with pretrained models. Each word occupies a line.
- max_seq_length: int type, default as 128. The maximum sequence length of tokens. Sequences whose length exceeds the value will be truncated.
- label_size: int type, default as None. The dimension of outputs, e.g. binary classification corresponds to label_size of 2. If set to None, the module will recognize and set the label size of training data.
- init_checkpoint: str type, default as None. The path or directory of checkpoint file to initialize. If set to None, parameters will be randomly initialized.
- output_dir: str type, default as None. The directory where module save outputs.
- gpu_ids: str or list type, default as None. The ids of GPUs that you wish to use in parallel computating (run `nvidia-smi` to check available GPUs). If set to None, tensorflow will select a single GPU to run.
- drop_pooler: bool type, default as False. BERT relevant argument that determines whether to ignore pooler layer in modeling.
- do_lower_case: bool type, default as True. Whether to convert inputs into lower case.
- truncated_method: str type, default as "LIFO". Three possible values are "Longer-FO" (longer-sequence-first-out), "LIFO" (last-in-first-out) and "FIFO" (first-in-first-out).

Check more information by running `help(XXX)` whenever you use new modules.

#### 1.2 Load module from configuration file

UNIF supports fast loading of module from configuration file. This is implemented by `.cache()` and `.load()` methods.

``` python
# cache configuration
model.cache('any code')

# load configuration
model = uf.BERTClassifier.load('any code')
```

Note: when attribute `output_dir`  is set to None, module saves configurations only, without saving graph into checkpoint.

### 2. Train/Predict/Score

Basic training, inference and scoring methods work similar with Scikit-Learns, implemented by `.fit()`, `.predict()` and `.score()` methods.

``` python
import uf

model = uf.BERTClassifier(config_file='demo/bert_config.json', vocab_file='demo/vocab.txt')

help(model.fit)
# uf.core.fit(X=None, y=None, sample_weight=None, X_tokenized=None, batch_size=32, learning_rate=5e-05, target_steps=None, total_steps=-3, warmup_ratio=0.1, print_per_secs=60, save_per_steps=1000, **kwargs)

help(model.predict)
# uf.core.predict(X=None, X_tokenized=None, batch_size=8)

help(model.score)
# uf.core.score(X=None, y=None, sample_weight=None, X_tokenized=None, batch_size=8)
```

- X, X_tokenized: list type, default as None. Either one should be None while the other is not. Feed tokenized inputs to X_tokenized if you wish to use your own tokenization tools. Each element correspond to one example.
- y: list type, default as None. Each element correspond to one example. y should be element-wisely alligned with X/X_tokenized.
- sample_weight: list type, default as None. The loss weight of samples. If set to None, each sample has the same loss weight, 1.
- batch_size: int type, default as 32. Training is generally faster if batch_size is set to a large value, but might result in OOM (out-of-memory).
- learning_rate: float type, default as 5e-5. A key argument who influences the overall performance of modules. Values from 1e-5 to 2e-4 are recommended to try. The larger model size, the lower learning rate.
- target_steps: int type, default as None. Breakpoint of training, where you wish to stop earlly and work on other fairs, e.g. scoring to measure model performance. If set to a positive value, then it is the stop point; if set to a negative value, then the training stops at `|target_steps|` epoches; if set to None, the training will run to the end.
- total_steps: int type, default as -3. Number of steps of the whole training life. If set to a positive value, then it is the total steps; if set to a negative value, then the training lasts `|total_steps|` epoches; if set to None, the training run 3 epoches.
- warmup_ratio: float type, default as 0.1. The learning rate warms up at first `warmup_ratio` steps and decays at 0.01 afterwards. The strategy is well known as slanted learning rate.
- print_per_secs: int type, default as 0.1. How many seconds to print training information.
- save_per_steps: int type, default as 1000. How many steps to save graph into checkpoint.

`kwargs` servers for other conditional arguments, e.g. `adversarial`, `epsilon` and `breg_miu` for adversarial training. See section 2.3 for more details.

#### 2.1 Optimizer

Four commonly used optimizers are supported in UNIF: GD, Adam, AdamW and LAMB. The default optimizer is AdamW. Feed value to argument `optimizer` when you wish to use others.

```python
model.fit(X, y, ..., optimizer='lamb')
```

#### 2.2 Layer-wise learning rate decay

LLRD is a valid trick to stabalize the convergence of neural networks. In reality, it improves performance simultaneously. A recommended value is 0.85. To use the trick, feed value to another argument `layerwise_lr_decay_ratio`.

``` python
model.fit(X, y, ..., layerwise_lr_decay_ratio=0.85)
```

A simple way to adjust the learning rate of a specific layer is to modify the layer power in `model._key_to_depths`. The deeper the layer, the lower the learning rate. Most module APIs support LLRD, except for some special ones, e.g. distillation models. When `model._key_to_depths` returns `'unsupported'`, the module does not support LLRD.

#### 2.3 Adversarial training

Adversarial training servers as another effective trick to fight against over-fitting and strengthen generalization. UNIF provides five choices: FGM, PGD, FreeLB, FreeAT and SMART.

``` python
# FGM
# accessible for most modules
model.fit(X, y, ..., adversarial='fgm', epsilon=0.5)

# PGD
# accessible for most modules
model.fit(X, y, ..., adversarial='pgd', epsilon=0.05, n_loop=2)

# FreeLB
# accessible for most modules
model.fit(X, y, ..., adversarial='freelb', epsilon=0.3, n_loop=3)

# FreeAT
# accessible for most modules
model.fit(X, y, ..., adversarial='freeat', epsilon=0.001, n_loop=3)

# SMART
# accessible only for classifier modules
model.fit(X, y, ..., adversarial='smart', epsilon=0.01, n_loop=2, prtb_lambda=0.5, breg_miu=0.2, tilda_beta=0.3)
```

Note: some modules is incompatible with adversarial training. Remove the arguments when meet errors.

#### 2.4 Large-batch training (via TFRecords)

Say if you wish to train a language model on large corpus, you can cache the converted training data into local TFRecords before you train, instead of crushing them into RAM. When you have a powerful multi-core CPU, it helps to train faster.

``` python
# cache data (auto)
# When `tfrecords_file` is None, create "tf.records" in `output_dir`.
model.to_tfrecords(X, y)

# training (auto)
# When `tfrecords_file` is None, read from "tf.records" in `output_dir`.
model.fit_from_tfrecords()

# training (manually)
# Reading from multiple tfrecords files is supported.
model.fit_from_tfrecords(tfrecords_files=['./tf.records', './tf.records.1'], n_jobs=3)

# To see the whole arguments.
help(model.fit_from_tfrecords)
# uf.core.fit_from_tfrecords(batch_size=32, learning_rate=5e-05, target_steps=None, total_steps=-3, warmup_ratio=0.1, print_per_secs=0.1, save_per_steps=1000, tfrecords_files=None, n_jobs=3, **kwargs)
# `n_jobs` is the number of threadings, default as 3
```

You can only write tfrecords into one file at one time, but can load data from multiple tfrecords files when training. For this reason, we suggest you to write tfrecords into more but smaller files before training and start fitting the model by setting more threadings.

Note: inference or scoring from tfrecords, e.g.  `.predict_from_tfrecords()`, is temporarily not supported. You can split the data into pieces and concatenate the outputs of inference.

### 3. Transfer learning

If you already have a pretrained checkpoint but fail to load into UNIF modules, a simple way can help if the failing is owed to the difference in name scope.

```python
# Check fail loaded variables.
print(model.uninited_vars)

# Check corresponding arrays in the checkpoint file.
tf.train.list_variables(model.init_checkpoint)

# Manually match the variable and array.
model.assignment_map['var_1_in_ckpt'] = model.uninited_vars['var_1_in_model']
model.assignment_map['var_2_in_ckpt'] = model.uninited_vars['var_2_in_model']

# Reload from checkpoint file.
model.reinit_from_checkpoint()

# See whether the fail loaded variables disappears from the list.
print(model.uninited_vars)

# Save graph into new checkpoint file in order not to repeat the above steps next time.
assert model.output_dir is not None
model.cache('any code')
```

### 4. TFServing

All module APIs support exporting into PB files for service deployment. You can check the interface information from printing.

``` python
# Export PB file into `output_dir`
assert model.output_dir is not None
model.export()
```

### 5. FAQ

- Q: What if I wish to take multiple segments as inputs?

  A: Use list to combine segments, e.g. `X = [['doc1 sent1', 'doc1 sent2', 'doc1 sent3'], ['doc2 sent1', 'doc2 sent2']]`. The module will recognize and process the segments.

- Q: How to check tokenization results?

  A: Run `model.tokenizer.tokenize(text)` to see how the module tokenize a text, or run `model.convert(X)` to capture the converted structured data.

- Q: What if I wish to use my own tokenization tools?

  A: Feed inputs to `X_tokenized` instead of `X`. Of course you need to tokenize the inputs in advance. Replace the former string input with a list of tokens.

- Q: Any suggestions of distillation?

  A: Fit `TinyBERTClassifier` and run `model.to_bert()` to save renamed parameters. Then implement secondary distillation by fitting `FastBERTClassifier`. Twice distillation helps you to achieve the maximum efficiency of inference.

- Q: Will UNIF supports Tensorflow 2.x?

  A: Tensorflow 2.x removed API contrib, which is a fairly import module to implement complex algorithms like CRF. Besides, the design of API usage is somewhat massive in tensorflow 2.x and bugs occurs without accessible solutions. The time when we support tensorflow 2.x depends on the acceptance level of worldwide tf-users.