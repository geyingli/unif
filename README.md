中文 | [English](./README_EN.md)

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
        <img src="https://img.shields.io/badge/version-beta2.4.13-blue">
    </a>
    <a>
        <img src="https://img.shields.io/badge/tensorflow-1.x 2.x-yellow">
    </a>
    <a>
        <img src="https://img.shields.io/badge/license-Apache2.0-red">
    </a>
</p>

有数据，想要快速实现你的想法？轻便、易使用的自然语言处理联合框架，支持快速搭建各类常用深度学习模型 (Transformer, GPT-2, BERT, RoBERTa, ALBERT, UniLM, ELECTRA)，同时对于 BERT 系列，支持高效用的蒸馏 (TinyBERT, FastBERT)。支持各类上下游任务 (语言模型、文本分类、文本生成、命名实体识别、机器阅读理解、机器翻译、序列标注等)。

### # 重要变更 (January 12, 2021)

从现版本起，通过 `.cache()` 将模型参数存储在配置文件后，可从任意工作目录下读取该模型，不再仅限于单一的工作目录。但由于本次功能改进，从前版本写入的配置文件读取模型可能会失效。

### 特性

- 仿 Scikit-Learn API 设计，三行代码完成训练及推理
- 同时支持 Tensorflow 1.x 及 2.x 版本
- 支持快速迁入已训练完毕的模型
- 支持对抗式训练等多项通用训练技巧
- 一键设置多 GPU 并行
- 导出 SavedModel，供部署使用
- 易开发，易扩展，能高效支持更多 State-Of-The-Art 算法

### 安装

在安装此依赖库之前需要预先安装 Python 3.6+ 及 Tensorflow 1.11+/2.x 版本。如果需要使用 GPU，请预先安装 Tensorflow GPU 版，并根据版本安装指定的英伟达 CUDA 工具包 (包括 cuDNN)，配置运行环境。例如 tensorflow-gpu\=\=1.11.0 对应 cuda\=\=9.0。

``` bash
git clone https://github.com/geyingli/unif
cd unif
python3 setup.py install
```

如果安装未获得权限，尝试 `python3 setup.py install --user`。

### 快速上手

来看看如何在数行代码之内完成训练和推理。这里我们提供了必备的配置文件作为 demo，无需提前下载预训练模型包。在刚才的安装目录下输入 `python3` 指令进入 Python 交互界面。

``` python
import uf

# 许可日志打印基本信息
uf.set_verbosity()

# 载入模型（使用 demo 配置文件进行示范）
model = uf.BERTClassifier(config_file='demo/bert_config.json', vocab_file='demo/vocab.txt')

# 定义训练样本
X, y = ['久旱逢甘露', '他乡遇故知'], [1, 0]

# 训练
model.fit(X, y)

# 推理
print(model.predict(X))
```

### API

| 应用 | API 				| 说明                                                 |
| :---------- | :----------- | :----------- |
| 语言模型 | `BERTLM` 		| 结合 MLM 和 NSP 任务，随机采样自下文及其他文档 |
|  		| `RoBERTaLM` 		| 单独 MLM 任务，采样至填满 max_seq_length |
|  		| `ALBERTLM` 		| 结合 MLM 和 SOP 任务，随机采样自上下文及其他文档 |
|  		| `ELECTRALM` 		| 结合 MLM 和 RTD 任务，生成器与判别器联合训练 |
|       | `VAELM` | 可生成语言文本负样本，也可提取向量用于聚类 |
|  | `GPT2LM` | 自回归式文本生成 |
|       | `UniLM` | 结合双向、单向及 Seq2Seq 任务的语言模型 |
| 命名实体识别 | `BERTNER` 		| 通过 BIESO 标签推导实体 |
|  		| `BERTCRFNER` 		| 基于维特比解码，通过 BIESO 标签推导实体 |
|  | `BERTCRFCascadeNER` | 基于维特比解码，通过 BIESO 标签推导实体；<br>识别实体的同时对实体进行分类 |
| 机器翻译 | `TransformerMT` | 共享词表，序列到序列建模 |
| 机器阅读理解 | `BERTMRC` 		| 从输入中抽取一段完整的答案 |
|  		| `RoBERTaMRC` 		| 从输入中抽取一段完整的答案 |
|  		| `ALBERTMRC` 		| 从输入中抽取一段完整的答案 |
|  		| `ELECTRAMRC` 		| 从输入中抽取一段完整的答案 |
|  		| `SANetMRC` 		| 从输入中抽取一段完整的答案 |
|  | `BERTVerifierMRC` | 从输入中抽取一段完整的答案，通过判题器辅助判断是否可答 |
|  | `RetroReaderMRC` | 从输入中抽取一段完整的答案，通过判题器辅助判断是否可答 |
| 单标签分类 | `TextCNNClassifier` 		| 每一个样本归属于一个唯一的类别 |
|  		| `BERTClassifier` 		| 每一个样本归属于一个唯一的类别 |
|  		| `XLNetClassifier` 		| 每一个样本归属于一个唯一的类别 |
|  		| `RoBERTaClassifier` 		| 每一个样本归属于一个唯一的类别 |
|  		| `ALBERTClassifier` 		| 每一个样本归属于一个唯一的类别 |
|  		| `ELECTRAClassifier` 		| 每一个样本归属于一个唯一的类别 |
|  		| `WideAndDeepClassifier` 		| 每一个样本归属于一个唯一的类别；<br>通过 Wide&Deep 结构结合更多句子级别的特征 |
|  		| `SemBERTClassifier` 		| 每一个样本归属于一个唯一的类别；<br>通过 SemBERT 结构结合更多字级别的特征 |
|  		| `PerformerClassifier` 		| 每一个样本归属于一个唯一的类别；<br>通过 FAVOR+ 加速推理 |
|  		| `UDAClassifier` 		| 融合一致性学习的半监督学习算法 |
| 多标签分类 | `BERTBinaryClassifier` 		| 每一个样本可同时属于零个或多个类别 |
|  		| `XLNetBinaryClassifier` 		| 每一个样本可同时属于零个或多个类别 |
|  		| `RoBERTaBinaryClassifier` 		| 每一个样本可同时属于零个或多个类别 |
|  		| `ALBERTBinaryClassifier` 		| 每一个样本可同时属于零个或多个类别 |
|  		| `ELECTRABinaryClassifier` 		| 每一个样本可同时属于零个或多个类别 |
| 序列标注 | `BERTSeqClassifier` 		| 每一个 token 都有唯一的类别 |
|  		| `XLNetSeqClassifier` 		| 每一个 token 都有唯一的类别 |
|  		| `RoBERTaSeqClassifier` 		| 每一个 token 都有唯一的类别 |
|  		| `ALBERTSeqClassifier` 		| 每一个 token 都有唯一的类别 |
|  		| `ELECTRASeqClassifier` 		| 每一个 token 都有唯一的类别 |
| 模型蒸馏 | `TinyBERTClassifier` 		| 大幅度压缩模型参数 |
|  		| `FastBERTClassifier` 		| 动态推理，易分样本提前离开模型 |

更多的使用细节，包括条件参数，可通过 `help(XXX)` 来查看，例如 `help(uf.BERTCRFCascadeNER)`。

### 模型搭建

#### 创建新模型

不同的模型拥有不同的参数需求，以 demo 为例：

```python
import uf

model = uf.BERTClassifier(config_file='demo/bert_config.json', vocab_file='demo/vocab.txt')
print(model)
# uf.BERTClassifier(config_file, vocab_file, max_seq_length=128, label_size=None, init_checkpoint=None, output_dir=None, gpu_ids=None, drop_pooler=False, do_lower_case=True, truncate_method='LIFO')
```

- config_file: str 类型，必填。指向与预训练参数配套的模型配置文件
- vocab_file: str 类型，必填。指向与预训练参数配套的词汇表，每一行代表一个词汇
- max_seq_length: int 类型，默认为 128。代表输入序列的最大长度，如果输入文本切词后超过该长度，则会被截断
- label_size: int 类型，默认为 None。分类器输出概率分布的维度，二分类时为 2，三分类时为 3，以此类推。如果为空，则会在训练时根据输入的 `y` 自动赋予合适的值
- init_checkpoint: str 类型，默认为 None。指向预训练参数，如果为空，则所有参数随机初始化
- output_dir: str 类型，默认为 None。模型参数及其他文件的保存地址
- gpu_ids: str 类型或 list 类型，默认为 None。当前设备可用 GPU 的 ID，可通过 `nvidia-smi` 指令查看。如果为空，则 tensorflow 自动寻找最合适的单 GPU 进行构图和运行
- drop_pooler: bool 类型，默认为 False。与 BERT 模型相关的参数，如果设为 True，则会在建模时忽略 pooler 层
- do_lower_case: bool 类型，默认为 True。切词器是否对样本进行小写处理
- truncated_method: str 类型，默认为 "LIFO"。三种可选取值为 "Longer-FO"、"LIFO" 和 "FIFO"，分别代表优先截断最长的序列、优先截断尾部词汇及优先阶段头部词汇

任何模型都可通过 `help(XXX)` 查看说明文档，了解参数含义。

#### 从配置快速读取模型

``` python
# 缓存配置
model.cache('代号', cache_file='.cache')

# 从配置读取模型
model = uf.load('代号', cache_file='.cache')
```

注：当模型的 `output_dir` 属性为空时，模型仅保存配置，不保存参数。

#### 重置

清空计算图，释放内存。

```python
model.reset()
```

### 训练/推理/评分

基础的训练、推理及评分的使用方法与 Scikit-Learn 十分相似，由 `.fit()`、`.predict()`和 `.score()` 来承载。

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

- X, X_tokenized: list 类型，默认为 None。二者必有一方为非空，如果希望使用模型自带的切词器，则为 X 赋值；如果希望使用自己的分词工具，则输入已经经过分词的语料，为 X_tokenized 赋值
- y: list 类型，默认为 None。如果模型的训练非监督学习，则无需赋值
- sample_weight: list 类型，默认为 None。样本权重，如果为空，则所有样本使用 1.0 的默认权重值
- batch_size: int 类型，默认为 32。适当增大有利于高效训练，但过高则会导致显存溢出
- learning_rate: float 类型，默认为 5e-5。影响模型表现的关键参数，推荐对以下取值均进行尝试，5e-5、1e-4、2e-4。通常参数规模越大，学习率越低
- target_steps: int 类型，默认为 None。预先设定的训练中断点，可在训练进行时跳出进程，进行验证集的评测。如果为正数，则直接设定为目标步数；如果为空，则跑完 total_steps；如果为负，则训练至 `|target_steps|` 轮，默认值代表训练三轮
- total_steps: int 类型，默认为 -3。总训练步数，影响学习率的变化。如果为正数，则直接设定为步数上限；如果为空，则自动根据 X 的大小和 batch_size 选择合适的 steps 数量，训练一轮；如果为负，则训练 `|total_steps|` 轮，默认值代表训练三轮
- warmup_ratio: float 类型，默认为 0.1。总步数的前一定比例采用线性增长的学习率，直到达到 learning_rate 的上限值
- print_per_secs: int 类型，默认为 0.1。每一定秒数打印一次训练信息
- save_per_steps: int 类型，默认为 1000。每一定步数保存一下 checkpoint 文件到 output_dir 目录

kwargs 承载其他模型特定的条件参数，例如在 `BERTClassifier` 的训练上使用对抗式训练算法，传入 `adversarial`、`epsilon`、`breg_miu` 等参数。详情见下文，第 2.3 小节。以下提供了一个“边训练边推理”的代码示例：

```python
# 假设你需要中途进行十次模型验证，而训练总共持续 6 轮
for loop_id in range(10):
    print('*** Running training on loop %d ***' % loop_id)
    model.fit(X, y, target_steps=((loop_id + 1) * -0.6), total_steps=-6)
    print('*** Running evaluation ***')
    print(model.score(X_dev, y_dev))
```

#### 优化器

框架目前支持四类优化器，GD、Adam、AdamW 和 LAMB，使用时在 `fit` 方法额外传入 `optimizer` 参数即可。

```python
model.fit(X, y, ..., optimizer='lamb')
```

#### 分层学习率

分层学习率是保证深度神经网络稳定收敛非常重要的一项 trick，在最终的评测表现上也有获得 0.5 以上百分比稳定的提升。建议取值为 0.85。同样作为 `fit` 的额外参数传入。

``` python
model.fit(X, y, ..., layerwise_lr_decay_ratio=0.85)
```

如果需要修改特定层对应的学习率，则可以通过 `model._key_to_depths` 来控制。`model._key_to_depths` 为一个字典，key 为变量关键词，value 为学习率衰减指数。层数越深，则学习率越小，层数最浅的一层学习率等于模型当前步长的学习率。

此外，需要注意的是，少部分模型不支持分层学习率 (例如蒸馏模型)。当 `model._key_to_depths` 的返回值为 `'unsupported'` 时，则表示该模型不支持分层学习率。

#### 对抗式训练

对抗式训练算法是非常有效的抗过拟合及增强泛化能力的训练技巧，在各类场景下都推荐进行尝试。UNIF 支持五类对抗式训练算法。

``` python
# 大多数类可用
model.fit(X, y, ..., adversarial='fgm', epsilon=0.5)    # FGM
model.fit(X, y, ..., adversarial='pgd', epsilon=0.05, n_loop=2)    # PGD
model.fit(X, y, ..., adversarial='freelb', epsilon=0.3, n_loop=3)    # FreeLB
model.fit(X, y, ..., adversarial='freeat', epsilon=0.001, n_loop=3)    # FreeAT

# 仅Classifier类可用
model.fit(X, y, ..., adversarial='smart', epsilon=0.01, n_loop=2, prtb_lambda=0.5, breg_miu=0.2, tilda_beta=0.3)    # SMART
```

注：部分模型与部分对抗式训练算法不兼容，如遇报错，可尝试其他对抗式训练算法。

#### 置信度过滤

当样本在任何一类的置信度大于阈值时，样本不参与损失的计算，由此避免过拟合，并使模型能更好地应付困难样本。目前仅单标签分类模块支持置信度过滤。

```python
model.fit(X, y, ..., conf_thresh=0.99)
```

#### 大批量训练 (via TFRecords)

当训练数据总数过大至内存难以承载时，可以考虑通过 `.to_tfrecords()` 将模型处理过后的训练数据缓存到本地，减轻训练时的内存负担。

``` python
# 缓存数据（自动创建文件）
# `tfrecords_file` 传入参数为空时，则默认在 `output_dir` 下创建名为 ".tfrecords" 的文件
model.to_tfrecords(X, y)

# 训练（自动指定文件）
# `tfrecords_files` 传入参数为空时，则默认取 `output_dir` 下名为 ".tfrecords" 的文件
model.fit_from_tfrecords()

# 训练（手动指定文件，可多个）
model.fit_from_tfrecords(tfrecords_files=['./.tfrecords', './.tfrecords.1'], n_jobs=3)

# 全部参数
help(model.fit_from_tfrecords)
# uf.core.fit_from_tfrecords(batch_size=32, learning_rate=5e-05, target_steps=None, total_steps=-3, warmup_ratio=0.1, print_per_secs=0.1, save_per_steps=1000, tfrecords_files=None, n_jobs=3, **kwargs)
# `n_jobs` 为线程数，默认为 3
```

缓存时仅能指定一个文件地址进行写入，但读取时可以同时读取多个文件，因此建议在进行大批量训练时（例如语言模型），分批将数据写入不同文件，而后多线程同时读取进行训练。

注：目前暂不支持预先缓存推理数据，可以通过分批推理完成批量推理任务。

### 迁移学习

#### 加载模型

如果已有从其他框架处训练完成的 checkpoint，发现由于参数命名不同而无法完全加载，则可通过以下方法解决。

```python
# 查看从 `init_checkpoint` 初始化失败的变量
assert model.init_checkpoint is not None
model.init()
print(model.uninited_vars)

# 在 `checkpoint` 中寻找对应的参数名
print(uf.list_variables(model.init_checkpoint))

# 将参数名与对应的被加载变量人工添加到 `assignment_map`
model.assignment_map['var_1_in_ckpt'] = model.uninited_vars['var_1_in_model']
model.assignment_map['var_2_in_ckpt'] = model.uninited_vars['var_2_in_model']

# 重新读取预训练参数
model.reinit_from_checkpoint()

# 查看变量是否从初始化失败的名单中消失
print(model.uninited_vars)

# 保存正确初始化的参数及配置（建议执行，否则下次载入模型需要重复上述步骤）
assert model.output_dir is not None
model.cache('代号')
```

#### 参数赋值

如果以上方式不能解决你的问题，可以通过赋值的方式达成加载参数的目的。

```python
import numpy as np

# 获取模型的引用地址
variable = model.trainable_variables[0]

# 赋值（这里使用全零张量为例）
shape = variable.shape.as_list()
value = np.zeros(shape)
assign_op = uf.tools.tf.assign(variable, value)
model.sess.run(assign_op)

# 查看是否赋值成功
print(model.sess.run(variable))

# 保存赋值后的参数及配置
assert model.output_dir is not None
model.cache('代号')
```

### TFServing

所有模型都支持导出 PB 文件，供模型上线，可以通过打印信息看到接口的定义及格式要求。

``` python
# 导出 PB 文件到 `output_dir` 下
assert model.output_dir is not None
model.export()
```

### FAQ

- 问：如何实现多个 segment 的输入？

  答：使用 list 组合多个 segment 的输入，如 `X = [['文档1句子1', '文档1句子2', '文档1句子3'], ['文档2句子1', '文档2句子2']]`，模型会自动按顺序拼接并添加分隔符。

- 问：如何查看切词结果？

  答：所有的模型切词工具统一使用 `.tokenizer` 承载。通过 `model.tokenizer.tokenize(text)` 可查看切词结果。另外也可通过 `model.convert(X)` 查看切词与 ID 转换后的矩阵。

- 问：如何使用自己的切词工具？

  答：在训练和推理时预先将传入参数 `X` 改为 `X_tokenized`，模型将直接跳过原有的的分词步骤。需要注意的是，分词结果同样需要基于 `list` 承载，例如原先由 `x` 承载的 `['黎明与夕阳']`，由 `X_tokenized` 承载后需呈现 `['黎', '##明', '与', '夕', '##阳']` 的形式。

- 问：如何实现 TinyBERT 和 FastBERT 复蒸馏？

  答：`TinyBERTClassifier` 训练完成后使用 `.to_bert()` 将变量重命名保存，而后使用 `FastBERTClassifier` 常规读取 checkpoint 和配置文件即可。

- 问：联合框架上一版本的 mixout、abandon_cls_info 等 tricks 为何不再使用？

  答：这些训练技巧目前证实无法帮助模型取得稳定增益。为了保持 API 尽量简单易用，遂将这些方法删除，未来可能通过其他方式重新加入回来。

### 需知

目前框架依然在开发和测试阶段，任何需求和建议都不尽欢迎。最后，感谢你读到这里。
