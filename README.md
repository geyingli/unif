[中文](./README_CN.md) | English

<p align="center">
    <br>
    	<img src="./docs/logo.png" style="zoom:70%"/>
    <br>
<p>
<p align="center">
    <a>
        <img src="https://img.shields.io/badge/build-passing-brightgreen">
    </a>
    <a>
        <img src="https://img.shields.io/badge/version-beta2.4.5-blue">
    </a>
    <a>
        <img src="https://img.shields.io/badge/tensorflow-1.x 2.x-yellow">
    </a>
    <a>
        <img src="https://img.shields.io/badge/license-Apache2.0-red">
    </a>
</p>

Wish to implement your ideas immediately? UNIF, as a unified language processing framework, supports building deep learning models in a simple and efficient manner, including Transformer, GPT-2, BERT, RoBERTa, ALBERT, XLNet, ELECTRA and etc. For BERT-series models, you need nothing but a single hot key to distill the model for light usage. Feel free to run applications among language modeling, text classification, text generation, named entity recognition, machine reading comprehension, machine translation and sequence labeling.

### Features

- Scikit-Learn like API design, train and predict in 3 lines of code
- Compatible with Tensorflow 1.x and 2.x
- Supports transfering pretrained model
- Support multiple deep learning tricks, e.g. adversarial training
- Multi-GPU parallelization
- Export SavedModel for industrial deployment
- Easy to develop and extend, chase after state-of-the-art algorithms

### Installation

Python 3.6+ and Tensorflow 1.11+/2.x are required to install the repo. If you with to run the models on GPU, please install NVIDIA CUDA toolkit in advance (be careful on the selection of published version).

``` bash
git clone https://github.com/geyingli/unif
cd unif
python3 setup.py install
```

If the installation is not authorized, try `python3 setup.py install --user`.

### Quick Tour

See how we train and predict in just several lines. Since we provide demo configuration files, no pretrained model is required. Input `python3` in command line and enter the interactive Python interface.

``` python
import uf

# allow printing basic information
uf.set_verbosity()

# load model (using demo files)
model = uf.BERTClassifier(config_file='demo/bert_config.json', vocab_file='demo/vocab.txt')

# define training samples
X, y = ['Natural language processing', 'is the core of AI.'], [1, 0]

# training
model.fit(X, y)

# inference
print(model.predict(X))
```

### API

| Application | API 				| Description                                       |
| :---------- | :----------- | :----------- |
| Language Modeling | `BERTLM` 		| Combine MLM and NSP task, sample sentences from context or other documents |
|  		| `RoBERTaLM` 		| Single MLM task, sample sentences to maximum sequence length |
|  		| `ALBERTLM` 		| Combine MLM and SOP task, sample sentences from context or other documents |
|  		| `ELECTRALM` 		| Combine MLM and RTD task, train generator and discriminator together |
|  | `GPT2LM` | Auto-regressive text generation |
| Named Entity Recognition | `BERTNER` 		| Recognize entities through BIESO labels |
|  		| `BERTCRFNER` 		| Recognize entities through BIESO labels and Viterbi decoding |
|  | `BERTCRFCascadeNER` | Recognize and classify entities through BIESO labels and Viterbi decoding |
| Machine Translation | `TransformerMT` | Sequence in and sequence out, with shared vocabulary |
| Machine Reading Comprehension | `BERTMRC` 		| Extract answer span from inputs |
|  		| `RoBERTaMRC` 		| Extract answer span from inputs |
|  		| `ALBERTMRC` 		| Extract answer span from inputs |
|  		| `ELECTRAMRC` 		| Extract answer span from inputs |
|  		| `SANetMRC` 		| Extract answer span from inputs |
|  | `BERTVerifierMRC` | Extract answer span from inputs, with a verifier assisted to judge whether the question is answerable |
|  | `RetroReaderMRC` | Extract answer span from inputs, with a verifier assisted to judge whether the question is answerable |
| Single-label Classification | `TextCNNClassifier` 	 | Each sample belongs to one class |
|  		| `BERTClassifier` 		| Each sample belongs to one class |
|  		| `XLNetClassifier` 		| Each sample belongs to one class |
|  		| `RoBERTaClassifier` 		| Each sample belongs to one class |
|  		| `ALBERTClassifier` 		| Each sample belongs to one class |
|  		| `ELECTRAClassifier` 		| Each sample belongs to one class |
|  		| `BERTWideAndDeepClassifier` 		| Each sample belongs to one class. Combined with more features through Wide&Deep structure |
|  		| `PerformerClassifier` 		| Each sample belongs to one class. Accelerate inference with FAVOR+ |
| Multi-label Classification | `BERTBinaryClassifier` 		| Each sample belongs to zero or multiple classes |
|  		| `XLNetBinaryClassifier` 		| Each sample belongs to zero or multiple classes |
|  		| `RoBERTaBinaryClassifier` 		| Each sample belongs to zero or multiple classes |
|  		| `ALBERTBinaryClassifier` 		| Each sample belongs to zero or multiple classes |
|  		| `ELECTRABinaryClassifier` 		| Each sample belongs to zero or multiple classes |
| Sequence Labeling | `BERTSeqClassifier` 		| Each token belongs to one single class |
|  		| `XLNetSeqClassifier` 		| Each token belongs to one single class |
|  		| `RoBERTaSeqClassifier` 		| Each token belongs to one single class |
|  		| `ALBERTSeqClassifier` 		| Each token belongs to one single class |
|  		| `ELECTRASeqClassifier` 		| Each token belongs to one single class |
| Knowledge Distillation | `TinyBERTClassifier` 		| Support distillation for all BERTClassifier, RoBERTaClassifier and ELECTRAClassifier |
|  		| `FastBERTClassifier` 		| Support distillation for all BERTClassifier, RoBERTaClassifier and ELECTRAClassifier |

More details about the usage as well as positional arguments, check through `help(XXX)`, e.g. `help(uf.BERTCRFCascadeNER)`.

### Modeling

#### New module

Modules have diverse argument choices. Taking demo as example:

```python
import uf

model = uf.BERTClassifier(config_file='demo/bert_config.json', vocab_file='demo/vocab.txt')
print(model)
# uf.BERTClassifier(config_file, vocab_file, max_seq_length=128, label_size=None, init_checkpoint=None, output_dir=None, gpu_ids=None, drop_pooler=False, do_lower_case=True, truncate_method='LIFO')
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

#### Load module from configuration file

``` python
# cache configuration
model.cache('any code')

# load configuration
model = uf.load('any code')
```

Note: when attribute `output_dir`  is set to None, module saves configurations only, without saving graph into checkpoint.

#### Reset

Clear computation graph and release the used memory.

```python
model.reset()
```

### Train/Predict/Score

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

#### Optimizer

Four commonly used optimizers are supported in UNIF: GD, Adam, AdamW and LAMB. The default optimizer is AdamW. Feed value to argument `optimizer` when you wish to use others.

```python
model.fit(X, y, ..., optimizer='lamb')
```

#### Layer-wise learning rate decay

LLRD is a valid trick to stabalize the convergence of neural networks. In reality, it improves performance simultaneously. A recommended value is 0.85. To use the trick, feed value to another argument `layerwise_lr_decay_ratio`.

``` python
model.fit(X, y, ..., layerwise_lr_decay_ratio=0.85)
```

A simple way to adjust the learning rate of a specific layer is to modify the layer power in `model._key_to_depths`. The deeper the layer, the lower the learning rate. Most module APIs support LLRD, except for some special ones, e.g. distillation models. When `model._key_to_depths` returns `'unsupported'`, the module does not support LLRD.

#### Adversarial training

Adversarial training servers as another effective trick to fight against over-fitting and strengthen generalization. UNIF provides five choices.

``` python
# accessible for most modules
model.fit(X, y, ..., adversarial='fgm', epsilon=0.5)    # FGM
model.fit(X, y, ..., adversarial='pgd', epsilon=0.05, n_loop=2)    # PGD
model.fit(X, y, ..., adversarial='freelb', epsilon=0.3, n_loop=3)    # FreeLB
model.fit(X, y, ..., adversarial='freeat', epsilon=0.001, n_loop=3)    # FreeAT

# accessible only for classifier modules
model.fit(X, y, ..., adversarial='smart', epsilon=0.01, n_loop=2, prtb_lambda=0.5, breg_miu=0.2, tilda_beta=0.3)    # SMART
```

Note: some modules is incompatible with adversarial training. Remove the arguments when meet errors.

#### Task Signal Annealing (TSA)

TSA is a training trick proposed by Google in paper [*Unsupversed Data Augmentation for Consistency Training*](https://arxiv.org/abs/1904.12848), who ignores losses from samples that have larger classification confidence level than threshold, while others still take part in the convergence. This trick helps to prevent over-fitting and looking down upon hard examples. As now, only single-label classification modules support TSA.

```python
model.fit(X, y, ..., tsa_thresh=0.05)
```

#### Large-batch training (via TFRecords)

Say if you wish to train a language model on large corpus, you can cache the converted training data into local TFRecords before you train, instead of crushing them into RAM. When you have a powerful multi-core CPU, it helps to train faster.

``` python
# cache data (auto)
# When `tfrecords_file` is None, create ".tfrecords" in `output_dir`.
model.to_tfrecords(X, y)

# training (auto)
# When `tfrecords_file` is None, read from ".tfrecords" in `output_dir`.
model.fit_from_tfrecords()

# training (manually)
# Reading from multiple tfrecords files is supported.
model.fit_from_tfrecords(tfrecords_files=['./.tfrecords', './.tfrecords.1'], n_jobs=3)

# To see the whole arguments.
help(model.fit_from_tfrecords)
# uf.core.fit_from_tfrecords(batch_size=32, learning_rate=5e-05, target_steps=None, total_steps=-3, warmup_ratio=0.1, print_per_secs=0.1, save_per_steps=1000, tfrecords_files=None, n_jobs=3, **kwargs)
# `n_jobs` is the number of threadings, default as 3
```

You can only write tfrecords into one file at one time, but can load data from multiple tfrecords files when training. For this reason, we suggest you to write tfrecords into more but smaller files before training and start fitting the model by setting more threadings.

Note: inference or scoring from tfrecords, e.g.  `.predict_from_tfrecords()`, is temporarily not supported. You can split the data into pieces and concatenate the outputs of inference.

### Transfer learning

#### Load Checkpoint

If you already have a pretrained checkpoint, but fail to load into UNIF modules simply because they have different name scope, a straight method can help:

```python
# Check fail loaded variables.
print(model.uninited_vars)

# Check corresponding arrays in the checkpoint file.
print(uf.list_variables(model.init_checkpoint))

# Manually match the variable and array.
model.assignment_map['var_1_in_ckpt'] = model.uninited_vars['var_1_in_model']
model.assignment_map['var_2_in_ckpt'] = model.uninited_vars['var_2_in_model']

# Reload from checkpoint file.
model.init_from_checkpoint()

# See whether the fail loaded variables disappears from the list.
print(model.uninited_vars)

# Save graph into new checkpoint file in order not to repeat the above steps next time.
assert model.output_dir is not None
model.cache('any code')
```

#### Set value for parameters

Once the above method does work for any reason, you can directly assign values for parameters:

```python
import numpy as np

# Get the reference of variable.
variable = model.trainable_variables[0]

# Assign values (we take all-zero tensor as example).
shape = variable.shape.as_list()
value = np.zeros(shape)
assign_op = uf.tools.tf.assign(variable, value)
model.sess.run(assign_op)

# See thether the parameter is successfully assigned.
print(model.sess.run(variable))

# Save graph into new checkpoint file with configurations.
assert model.output_dir is not None
model.cache('代号')
```

### TFServing

All module APIs support exporting into PB files for service deployment. You can check the interface information from printing.

``` python
# Export PB file into `output_dir`
assert model.output_dir is not None
model.export()
```

### FAQ

- Q: What if I wish to take multiple segments as inputs?

  A: Use list to combine segments, e.g. `X = [['doc1 sent1', 'doc1 sent2', 'doc1 sent3'], ['doc2 sent1', 'doc2 sent2']]`. The module will recognize and process the segments.

- Q: How to check tokenization results?

  A: Run `model.tokenizer.tokenize(text)` to see how the module tokenize a text, or run `model.convert(X)` to capture the converted structured data.

- Q: What if I wish to use my own tokenization tools?

  A: Feed inputs to `X_tokenized` instead of `X`. Of course you need to tokenize the inputs in advance. Replace the former string input with a list of tokens.

- Q: Any suggestions of distillation?

  A: Fit `TinyBERTClassifier` and run `model.to_bert()` to save renamed parameters. Then implement secondary distillation by fitting `FastBERTClassifier`. Twice distillation helps you to achieve the maximum efficiency of inference.

### Acknowledgement

The repo's still in development and testing stage. Any suggestions are mostly welcome. At last, thanks for your interest.
