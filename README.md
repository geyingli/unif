[中文](./docs/README_CN.md) | English

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
        <img src="https://img.shields.io/badge/version-beta2.4.1-blue">
    </a>
    <a>
        <img src="https://img.shields.io/badge/tensorflow-1.x 2.x-yellow">
    </a>
    <a>
        <img src="https://img.shields.io/badge/license-Apache2.0-red">
    </a>
</p>

Wish to implement your ideas immediately? UNIF, as a unified language processing framework, supports building deep learning models in a simple and efficient manner, including Transformer, GPT-2, BERT, RoBERTa, ALBERT, XLNet, ELECTRA and etc. For BERT-series models, you need nothing but a single hot key to distill the model for light usage. Feel free to run applications among language modeling, text classification, text generation, named entity recognition, machine reading comprehension, machine translation and sequence labeling. UNIF is all you need.


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
python setup.py install
```

If the installation if not authorized, try `python setup.py install --user`.

### Quick Tour

See how we train and predict in just several lines. Since we provide demo configuration files, no pretrained model is required. Input `python` in command line and enter the interactive Python interface.

``` python
import uf

# allow printing basic information
uf.set_verbosity(2)

# load model (using demo files)
model = uf.BERTClassifier(config_file='demo/bert_config.json', vocab_file='demo/vocab.txt')

# define training samples
X, y = ['Natural language processing', 'is the core of AI.'], [1, 0]

# training
model.fit(X, y)

# inference
print(model.predict(X))
```

For FAQs and more instructions about building and training models, see [tutorial.md](./docs/tutorial.md).

### API

Available APIs are displayed as below. More details about the usage as well as positional arguments, check through `help(XXX)`, e.g. `help(uf.BERTCRFCascadeNER)`.

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

### Acknowledgement

The repo's still in development and testing stage. Any suggestions are mostly welcome. At last, thanks for your interest.
