[中文](README_CN.md) | [English](README.md)

<p align="center">
    <br>
    	<img src="logo.png" style="zoom:70%"/>
    <br>
<p>
<p align="center">
    <a>
        <img src="https://img.shields.io/badge/build-passing-brightgreen">
    </a>
    <a>
        <img src="https://img.shields.io/badge/version-beta2.1.25-blue">
    </a>
    <a>
        <img src="https://img.shields.io/badge/tensorflow-≥1.11.0-yellow">
    </a>
    <a>
        <img src="https://img.shields.io/badge/license-Apache2.0-red">
    </a>
</p>


Abundant data and ideas on hand but having no way to implement immediately? UNIF, as a unified language processing framework, supports building deep learning models in a simple and efficient manner, including Transformer, GPT-2, BERT, RoBERTa, ALBERT, XLNet, ELECTRA and etc. For BERT-series models, you need nothing but a single hot key to distill the model for light usage. Feel free to run applications among language modeling, text classification, text generation, named entity recognition, machine reading comprehension, machine translation and sequence labeling. UNIF is all you need.

### Features

- Scikit-Learn like API design, train and predict in 3 lines of code
- Supports transfering pretrained model
- Support multiple deep learning tricks, e.g. adversarial training
- Multi-GPU parallelization
- Export SavedModel for industrial deployment
- Easy to develop and extend, chase after state-of-the-art algorithms

### Install

Python 3.6+ and Tensorflow 1.11+ are required to install the repo. If you with to run the models on GPU, please install NVIDIA CUDA toolkit in advance (be careful on the selection of published version). 

``` bash
git clone https://github.com/geyingli/unif
cd unif
python setup.py install
```

If the installation if not authorized, try `python setup.py install --user`.

### Quick Tour

See how we train and predict in just several lines. Since we provide demo configuration files, no pretrained model is required. Input `ipython` in command line and enter the interactive Python interface.

``` python
import tensorflow as tf
import uf

# allow printing information
tf.logging.set_verbosity(tf.logging.INFO)

# load model (by using demo files)
model = uf.BERTClassifier(config_file='demo/bert_config.json', vocab_file='demo/vocab.txt')

# define training samples
X, y = ['Natural language processing', 'is the core of AI.'], [1, 0]

# training
model.fit(X, y)

# inference
print(model.predict(X))
```

For FAQs and more instructions about building and training models, see [tutorial.md](./tutorial.md).

### API

Available APIs include:

| API 				| 类别         | 说明                                                 |
| ----------- | ------------ | ---------------------------------------------------- |
| `BERTLM` 		| Language Modeling | Combine MLM and NSP task, sample sentences from context or other documents |
| `RoBERTaLM` 		| Language Modeling | Single MLM task, sample sentences to maximum sequence length |
| `ALBERTLM` 		| Language Modeling | Combine MLM and SOP task, sample sentences from context or other documents |
| `ELECTRALM` 		| Language Modeling | Combine MLM and RTD task, train generator and discriminator together |
| `GPT2LM` | Language Modeling | Auto-regressive text generation |
| `BERTNER` 		| Named Entity Recognition | Recognize entities through BIESO labels |
| `BERTCRFNER` 		| Named Entity Recognition | Recognize entities through BIESO labels and Viterbi decoding |
| `BERTCRFCascadeNER` | Named Entity Recognition | Recognize and classify entities through BIESO labels and Viterbi decoding |
| `TransformerMT` | Machine Translation | Sequence in and sequence out, with shared vocabulary |
| `BERTMRC` 		| Machine Reading Comprehension | Extract answer span from inputs |
| `RoBERTaMRC` 		| Machine Reading Comprehension | Extract answer span from inputs |
| `ALBERTMRC` 		| Machine Reading Comprehension | Extract answer span from inputs |
| `ELECTRAMRC` 		| Machine Reading Comprehension | Extract answer span from inputs |
| `BERTClassifier` 		| Single-label Classification | Each sample belongs to one class |
| `XLNetClassifier` 		| Single-label Classification | Each sample belongs to one class |
| `RoBERTaClassifier` 		| Single-label Classification | Each sample belongs to one class |
| `ALBERTClassifier` 		| Single-label Classification | Each sample belongs to one class |
| `ELECTRAClassifier` 		| Single-label Classification | Each sample belongs to one class |
| `BERTBinaryClassifier` 		| Multi-label Classification | Each sample belongs to zero or multiple classes |
| `XLNetBinaryClassifier` 		| Multi-label Classification | Each sample belongs to zero or multiple classes |
| `RoBERTaBinaryClassifier` 		| Multi-label Classification | Each sample belongs to zero or multiple classes |
| `ALBERTBinaryClassifier` 		| Multi-label Classification | Each sample belongs to zero or multiple classes |
| `ELECTRABinaryClassifier` 		| Multi-label Classification | Each sample belongs to zero or multiple classes |
| `BERTSeqClassifier` 		| Sequence Labeling | Each token belongs to one single class |
| `XLNetSeqClassifier` 		| Sequence Labeling | Each token belongs to one single class |
| `RoBERTaSeqClassifier` 		| Sequence Labeling | Each token belongs to one single class |
| `ALBERTSeqClassifier` 		| Sequence Labeling | Each token belongs to one single class |
| `ELECTRASeqClassifier` 		| Sequence Labeling | Each token belongs to one single class |
| `TinyBERTClassifier` 		| Knowledge Distillation | Support knowledge distillation for all single-label classifiers except XLNetClassifier |
| `FastBERTClassifier` 		| Knowledge Distillation | Support knowledge distillation for all single-label classifiers except XLNetClassifier |

### Performance on Public Leaderboards

...

### Development

...