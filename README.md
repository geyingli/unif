<p align="left">
    <br>
    	<img src="logo.png" style="zoom:100%"/>
    <br>
<p>
<p align="left">
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

有数据，想要快速实现你的想法？轻便、易使用的自然语言处理联合框架，支持快速搭建各类常用深度学习模型 (Transformer, GPT-2, BERT, RoBERTa, ALBERT, XLNet, ELECTRA)，同时对于 BERT 系列，支持高效用的蒸馏 (TinyBERT, FastBERT)。支持各类下游任务 (文本分类、文本生成、命名实体识别、机器阅读理解、机器翻译、序列标注等)。UniF is all you need。

### 特性

- 仿 SkLearn API 设计，三行代码完成训练及推理
- 支持快速迁入已训练完毕的模型
- 支持对抗式训练等多项通用训练技巧
- 支持一键设置多 GPU 并行
- 易开发，易扩展，能高效支持更多 State-Of-The-Art 算法

### 安装

在安装此依赖库之前需要预先安装 Python 3.6+ 及 Tensorflow 1.11+ 版本。如果需要使用 GPU，请预先根据 Tensorflow 版本，安装指定的英伟达 CUDA 工具包，配置运行环境。

``` bash
git clone https://git.code.oa.com/geyingli/uf
cd uf
python setup.py install
```

如果安装未获得权限，尝试 `python setup.py install --user`。

### 快速上手

来看看如何在数行代码之内完成训练和推理。这里我们提供了必备的配置文件作为 demo，无需提前下载预训练模型包。在刚才的安装目录下输入 `ipython` 指令进入 Python 交互界面。

``` python
import tensorflow as tf
import uf

# 许可日志打印
tf.logging.set_verbosity(tf.logging.INFO)

# 载入模型（使用 demo 配置文件进行示范）
model = uf.BERTClassifier(config_file='demo/bert_config.json', vocab_file='demo/vocab.txt')

# 定义训练样本
X, y = ['久旱逢甘露', '他乡遇故知'], [1, 0]

# 训练
model.fit(X, y)

# 推理
print(model.predict(X))
```

包括 FAQ 及建模、训练等进一步的指导和说明，见 [tutorial.md](./tutorial.md)。

### API

当前版本已开放的成熟 API 包括：

| API 				| 类别         | 说明                                                 |
| ----------- | ------------ | ---------------------------------------------------- |
| `BERTLM` 		| 语言模型 | 结合 MLM 和 NSP 任务，随机采样自下文及其他文档 |
| `RoBERTaLM` 		| 语言模型 | 单独 MLM 任务，采样至填满 max_seq_length |
| `ALBERTLM` 		| 语言模型 | 结合 MLM 和 SOP 任务，随机采样自上下文及其他文档 |
| `ELECTRALM` 		| 语言模型 | 结合 MLM 和 RTD 任务，生成器与判别器联合训练 |
| `GPT2LM` | 语言模型 | 自回归式文本生成 |
| `BERTNER` 		| 命名实体识别 | 通过 BIESO 标签推导实体 |
| `BERTCRFNER` 		| 命名实体识别 | 基于维特比解码，通过 BIESO 标签推导实体 |
| `BERTCRFCascadeNER` | 命名实体识别 | 基于维特比解码，通过 BIESO 标签推导实体；识别实体的同时对实体进行分类 |
| `TransformerMT` | 机器翻译 | 共享词表，序列到序列建模 |
| `BERTMRC` 		| 机器阅读理解 | 从输入中抽取一段完整的答案 |
| `RoBERTaMRC` 		| 机器阅读理解 | 从输入中抽取一段完整的答案 |
| `ALBERTMRC` 		| 机器阅读理解 | 从输入中抽取一段完整的答案 |
| `ELECTRAMRC` 		| 机器阅读理解 | 从输入中抽取一段完整的答案 |
| `BERTClassifier` 		| 单标签分类 | 每一个样本归属于一个唯一的类别 |
| `XLNetClassifier` 		| 单标签分类 | 每一个样本归属于一个唯一的类别 |
| `RoBERTaClassifier` 		| 单标签分类 | 每一个样本归属于一个唯一的类别 |
| `ALBERTClassifier` 		| 单标签分类 | 每一个样本归属于一个唯一的类别 |
| `ELECTRAClassifier` 		| 单标签分类 | 每一个样本归属于一个唯一的类别 |
| `BERTBinaryClassifier` 		| 多标签分类 | 每一个样本可同时属于零个或多个类别 |
| `XLNetBinaryClassifier` 		| 多标签分类 | 每一个样本可同时属于零个或多个类别 |
| `RoBERTaBinaryClassifier` 		| 多标签分类 | 每一个样本可同时属于零个或多个类别 |
| `ALBERTBinaryClassifier` 		| 多标签分类 | 每一个样本可同时属于零个或多个类别 |
| `ELECTRABinaryClassifier` 		| 多标签分类 | 每一个样本可同时属于零个或多个类别 |
| `BERTSeqClassifier` 		| 序列标注 | 每一个 token 都有唯一的类别 |
| `XLNetSeqClassifier` 		| 序列标注 | 每一个 token 都有唯一的类别 |
| `RoBERTaSeqClassifier` 		| 序列标注 | 每一个 token 都有唯一的类别 |
| `ALBERTSeqClassifier` 		| 序列标注 | 每一个 token 都有唯一的类别 |
| `ELECTRASeqClassifier` 		| 序列标注 | 每一个 token 都有唯一的类别 |
| `TinyBERTClassifier` 		| 模型蒸馏 | 支持蒸馏除 XLNetClassifier 以外的所有单标签分类器 |
| `FastBERTClassifier` 		| 模型蒸馏 | 支持蒸馏除 XLNetClassifier 以外的所有单标签分类器 |

### 公开榜单评测

十月底前推出。

### 开发需知

十月底前推出。