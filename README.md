中文 | [English](./README_EN.md)

<p align="center">
    <br>
    	<img src="./docs/logo.png" style="zoom:74%"/>
    <br>
<p>
<p align="center">
    <a>
        <img src="https://img.shields.io/badge/build-passing-brightgreen">
    </a>
    <a>
        <img src="https://img.shields.io/badge/version-beta v2.9.3-blue">
    </a>
    <a>
        <img src="https://img.shields.io/badge/tensorflow-1.x\2.x-yellow">
    </a>
    <a>
        <img src="https://img.shields.io/badge/license-Apache2.0-red">
    </a>
</p>


快速搭建各类深度学习模型 (Transformer, GPT-2, BERT, ALBERT, UniLM, XLNet, ELECTRA 等)，完成语言模型/文本分类/文本生成/命名实体识别/机器阅读理解/机器翻译/序列标注/知识蒸馏任务。

### 特性

- API 简单：三行代码完成训练及推理，并一键设置多进程/多 GPU 并行
- 品类丰富：支持约 40 种模型类
- 依赖简单：Tensorflow 1.x/2.x
- 高分保证：提供分层学习率、对抗式训练等多项训练技巧
- 可供部署：导出模型 PB 文件，供线上部署

### 安装

``` bash
git clone https://github.com/geyingli/unif
cd unif
python3 setup.py install --user
```

若需卸载，通过 `pip uninstall uf` 即可。

### 快速上手

``` python
import uf

# 新建模型（使用 demo 配置文件进行示范）
model = uf.BERTClassifier(config_file='./demo/bert_config.json', vocab_file='./demo/vocab.txt')

# 定义训练样本
X, y = ['久旱逢甘露', '他乡遇故知'], [1, 0]

# 训练
model.fit(X, y)

# 推理
print(model.predict(X))
```

## 模型列表

| 领域 | API 				| 简介                                               |
| :---------- | :----------- | :----------- |
| 语言模型 | [`BERTLM`](./examples/BERTLM.ipynb) | 结合 MLM 和 NSP 任务，随机采样自下文及其他文档 |
|  		| [`RoBERTaLM`](./examples/RoBERTaLM.ipynb) 		| 仅 MLM 任务，采样至文档结束 |
|  		| [`ALBERTLM`](./examples/ALBERTLM.ipynb) 		| 结合 MLM 和 SOP，随机采样自上下文及其他文档 |
|  		| [`ELECTRALM`](./examples/ELECTRALM.ipynb) 		| 结合 MLM 和 RTD，生成器与判别器联合训练 |
|       | [`VAELM`](./examples/VAELM.ipynb) | 可生成语言文本负样本，也可提取向量用于聚类 |
|  | [`GPT2LM`](./examples/GPT2LM.ipynb) | 自回归式文本生成 |
|       | [`UniLM`](./examples/UniLM.ipynb) | 结合双向、单向及 Seq2Seq 建模的全能语言模型 |
| 命名实体识别 | [`BERTNER`](./examples/BERTNER.ipynb) 		| - |
|  		| [`BERTCRFNER`](./examples/BERTCRFNER.ipynb) 		| 结合 CRF |
|  | [`BERTCRFCascadeNER`](./examples/BERTCRFCascadeNER.ipynb) | 识别与分类同时进行的级联架构 |
| 机器翻译 | [`TransformerMT`](./examples/TransformerMT.ipynb) | 共享词表，标准 Seq2Seq 架构 |
| 机器阅读理解 | [`BERTMRC`](./examples/BERTMRC.ipynb) 		| - |
|  		| [`RoBERTaMRC`](./examples/RoBERTaMRC.ipynb) 		| - |
|  		| [`ALBERTMRC`](./examples/ALBERTMRC.ipynb) 		| - |
|  		| [`ELECTRAMRC`](./examples/ELECTRAMRC.ipynb) 		| - |
|  		| [`SANetMRC`](./examples/SANetMRC.ipynb) 		| 引入 Sentence Attention |
|  | [`BERTVerifierMRC`](./examples/BERTVerifierMRC.ipynb) | 抽取 answer span 的同时判断可答性 |
|  | [`RetroReaderMRC`](./examples/RetroReaderMRC.ipynb) | 抽取 answer span 的同时判断可答性 |
| 单 Label 分类 | [`TextCNNClassifier`](./examples/TextCNNClassifier.ipynb) 		| 小而快 |
|  		| [`BERTClassifier`](./examples/BERTClassifier.ipynb) 		| - |
|  		| [`XLNetClassifier`](./examples/XLNetClassifier.ipynb) 		| - |
|  		| [`RoBERTaClassifier`](./examples/RoBERTaClassifier.ipynb) 		| - |
|  		| [`ALBERTClassifier`](./examples/ALBERTClassifier.ipynb) 		| - |
|  		| [`ELECTRAClassifier`](./examples/ELECTRAClassifier.ipynb) 		| - |
|  		| [`WideAndDeepClassifier`](./examples/WideAndDeepClassifier.ipynb) | 通过 Wide & Deep 架构融合句子级别特征 |
|  		| [`SemBERTClassifier`](./examples/SemBERTClassifier.ipynb) 		| 通过 SemBERT 架构融合字级别的特征 |
|  		| [`UDAClassifier`](./examples/UDAClassifier.ipynb) 		| 结合一致性学习的半监督学习算法 |
| 多 Label 分类 | [`BERTBinaryClassifier`](./examples/BERTBinaryClassifier.ipynb) 		| - |
|  		| [`XLNetBinaryClassifier`](./examples/XLNetBinaryClassifier.ipynb) 		| - |
|  		| [`RoBERTaBinaryClassifier`](./examples/RoBERTaBinaryClassifier.ipynb) 		| - |
|  		| [`ALBERTBinaryClassifier`](./examples/ALBERTBinaryClassifier.ipynb) 		| - |
|  		| [`ELECTRABinaryClassifier`](./examples/ELECTRABinaryClassifier.ipynb) 		| - |
| 序列标注 | [`BERTSeqClassifier`](./examples/BERTSeqClassifier.ipynb) 		| - |
|  		| [`RoBERTaSeqClassifier`](./examples/RoBERTaSeqClassifier.ipynb) 		| - |
|  		| [`ALBERTSeqClassifier`](./examples/ALBERTSeqClassifier.ipynb) 		| - |
|  		| [`ELECTRASeqClassifier`](./examples/ELECTRASeqClassifier.ipynb) 		| - |
| 模型蒸馏 | [`TinyBERTClassifier`](./examples/TinyBERTClassifier.ipynb) 		| 大幅压缩模型参数，提速十倍以上 |
|       | [`TinyBERTBinaryClassifier`](./examples/TinyBERTBinaryClassifier.ipynb)     | - |
|  		| [`FastBERTClassifier`](./examples/FastBERTClassifier.ipynb) 		| 动态推理，易分样本提前离开模型 |

点击上方模型名称，查看简要的使用示范。此外，善用  `help(XXX)` 能帮你获得更多 API 的使用细节。

## 建模

一步创建新模型：

```python
model = uf.BERTClassifier(
    config_file, vocab_file,
    max_seq_length=128,
    label_size=2,
    init_checkpoint=None,    # 预训练参数路径
    output_dir='./output',
    gpu_ids='0,1,3,5',
    drop_pooler=False,    # 建模时跳过 pooler 层
    do_lower_case=True,
    truncate_method='LIFO')    # longer-FO/LIFO/FIFO
```

下载知名公开预训练参数：

```python
uf.list_resources()    # 查看参数列表
uf.download('bert-wwm-ext-base-zh')    # 下载预训练模型包
```

任务后期需要大量的训练，可以通过配置文件，方便地管理模型：

``` python
# 写入配置文件
assert model.output_dir is not None    # 为空的话模型就白训了
model.cache('key', cache_file='.cache')

# 从配置文件读取
model = uf.load('key', cache_file='.cache')
```

模型使用完毕，想清出内存？试试 `del model` 或重置 `model.reset()`。

## 训练/推理/评分

``` python
# 开启多进程 (加速数据预处理)
with uf.MultiProcess():    # 多进程的本质是将当前进程进行复制，因此建议读取数据的代码写在这一步之后，否则容易内存爆炸
    X, y = load_data()

    # 训练
    model.fit(
        X=X, y=y,
        sample_weight=None,    # 样本权重，放空则默认每条样本权重为 1.0
        X_tokenized=None,    # 特定场景下使用，e.g. 使用你自己的分词工具/语言模型推理时在输入加 [MASK]
        batch_size=32,
        learning_rate=5e-05,
        target_steps=None,    # 放空代表直接训练到 `total_steps`，不中途停止；否则为本次训练断点
        total_steps=-3,    # -3 代表自动计算数据量并循环三轮
        warmup_ratio=0.1,
        print_per_secs=1,    # 多少秒打印一次信息
        save_per_steps=1000,
        **kwargs)    # 其他参数，下文介绍

    # 推理
    model.predict(
        X=X, X_tokenized=None,
        batch_size=8)

    # 评分
    model.score(
        X=X, y=y, sample_weight=None, X_tokenized=None,
        batch_size=8)

# 常规训练流程示范
assert model.output_dir is not None    # 非空才能保存模型参数
for loop_id in range(10):    # 假设训练途中一共验证 10 次
    model.fit(X, y, target_steps=((loop_id + 1) * -0.6), total_steps=-6)    # 假设一共训练 6 轮
    model.cache('dev-%d' % loop_id)    # 保存一次模型
    print(model.score(X_dev, y_dev))    # 查看模型表现
```

数据体量太大，爆内存？可以尝试先分批将数据写入不同的 TFRecords 文件，训练时同步读取：

```python
with uf.MultiProcess():

    # 缓存数据
    model.to_tfrecords(
        X=X, y=y, sample_weight=None, X_tokenized=None,
        tfrecords_file='./train.tfrecords')    # 一次只能存一个文件

# 边读边训
model.fit_from_tfrecords(
    tfrecords_files=['./train.tfrecords-0', './.tfrecords-1'],    # 同步从多个 TFRecords 文件读取
    n_jobs=3,    # 启动三个线程
    batch_size=32,    # 以下参数和 `.fit()` 中参数相同
    learning_rate=5e-05,
    target_steps=None,
    total_steps=-3,
    warmup_ratio=0.1,
    print_per_secs=1,
    save_per_steps=1000,
    **kwargs)
```

其他训练参数：

```python
# 优化器
model.fit(X, y, ..., optimizer='gd')
model.fit(X, y, ..., optimizer='adam')
model.fit(X, y, ..., optimizer='adamw')    # 默认
model.fit(X, y, ..., optimizer='lamb')

# 分层学习率：应对迁移学习中的 catastrophic forgetting 问题 (少量模型不适用)
model.fit(X, y, ..., layerwise_lr_decay_ratio=0.85)    # 默认为 None
print(model._key_to_depths)    # 衰减比率 (可手动进行修改，修改完成后训练即生效)

# 对抗式训练：在输入中添加扰动，以提高模型的鲁棒性和泛化能力
model.fit(X, y, ..., adversarial='fgm', epsilon=0.5)    # FGM
model.fit(X, y, ..., adversarial='pgd', epsilon=0.05, n_loop=2)    # PGD
model.fit(X, y, ..., adversarial='freelb', epsilon=0.3, n_loop=3)    # FreeLB
model.fit(X, y, ..., adversarial='freeat', epsilon=0.001, n_loop=3)    # FreeAT
model.fit(X, y, ..., adversarial='smart', epsilon=0.01, n_loop=2, prtb_lambda=0.5, breg_miu=0.2, tilda_beta=0.3)    # SMART (仅 Classifier 可用)

# 置信度过滤：样本置信度达到阈值后不再参与训练，避免过拟合 (仅 Classifier 可用)
model.fit(X, y, ..., conf_thresh=0.99)    # 默认为 None

# 梯度累积：当 batch_size 过小以至于模型拟合困难时，梯度累积可以显著提高拟合速度
model.fit(X, y, ..., grad_acc_steps=5)    # 默认为 1
```

## 迁移学习

存在变量命名不同而无法加载，可通过以下步骤解决：

```python
# 查看从 `init_checkpoint` 初始化失败的变量
assert model.init_checkpoint is not None
model.init()
print(model.uninited_vars)

# 人工构建变量名映射规则，重新读取变量
print(uf.list_variables(model.init_checkpoint))    # 在 `checkpoint` 中寻找对应的参数名
model.assignment_map['var_name_from_ckpt_file'] = model.uninited_vars['var_name_in_uf_model']    # 添加映射关系
model.reinit_from_checkpoint()    # 重新读取预训练参数
print(model.uninited_vars)    # 看看变量是否从初始化失败的名单中消失

# 保存参数及配置（避免下次载入预训练参数时，重复上述步骤）
assert model.output_dir is not None
model.cache('key')
```

直接给参数赋值如何？当然是可以的：

```python
variable = model.trainable_variables[0]    # 获取参数
model.assign(variable, value)    # 赋值
print(model.sess.run(variable))    # 查看参数

# 保存赋值后的参数及配置
assert model.output_dir is not None
model.cache('key')
```

## TFServing

导出 PB 文件到指定目录下：

``` python
model.export(
    'serving',    # 导出目录
    rename_inputs={},    # 重命名输入
    rename_outputs={},    # 重命名输出
    ignore_inputs=[],    # 剪裁多余输入
    ignore_outputs=[])    # 裁剪多余输出
```

## 开发需知

我们欢迎一切有效的 pull request，加入我们，成为 contributors 一员。 核心的代码架构如下图所示，新的模型开发仅需要在 application 下添加新的类，这些类可以由现有算法库 modeling 中的算子组合而来，也可以自行编写。

<p align="center">
    <br>
    	<img src="./docs/framework.png" style="zoom:50%"/>
    <br>
<p>

## FAQ

- 问：有什么提高训练速度的方法吗？

  答：首先是几种我们能立即实施的基础方法：减小 max_seq_length，多 GPU 并行，多进程数据处理，以及梯度累积。在这些之外，可以进一步尝试对输入的数据进行拆分，在训练过程中逐步提高 max_seq_length、batch_size 和 dropout_rate (通过提高拟合速度，缩短整个训练周期)。当然，还有一些在 UNIF 暂时无法实现的功能，可以前往其他 repo 寻求解决方案，包括但不限于混合精度训练、OP融合、使用 Linformer 等时间复杂度小于 O(N^2) 的模型。

- 问：训练时内存不足，该怎么办？

  答：首先需要明确是显存溢出还是内存爆炸。如果是显存溢出，则需要降低 batch_size；如果是由于数据体量过于庞大导致的内存爆炸，可以尝试通过 `model.to_tfrecords()` 分批将数据写入 TFRecords 文件，而后清出内存，通过 `model.fit_from_tfrecords()` 读取进行训练。

- 问：模型输入有什么限制吗？

  答：对于大多数模型来说，没有限制。一条样本，可以是一个字符串，也可以是多个字符串。以 BERT 为例，完全不必局限于一到两个句子的输入，而是可以通过 list 组合多个 segment，e.g.  `X = [['文档1句子1', '文档1句子2', '文档1句子3'], ['文档2句子1', '文档2句子2']]`，模型会自动按顺序拼接并添加分隔符。

- 问：如何查看切词结果？

  答：通过 `model.tokenizer.tokenize(text)` 可查看切词结果。另外也可通过 `model.convert(X)` 查看经过预处理的实际的模型输入。

- 问：如果我想使用自己的切词工具，该怎么做？

  答：提前使用自己的工具做好分词，而后在训练和推理时将原先的传入参数由 `X` 改为 `X_tokenized`。例如，原本传入 `model.fit(X=['黎明与夕阳', ...], ...)`，使用自己的分词工具后，改为传入 `model.fit(X_tokenized=[['黎', '##明', '与', '夕', '##阳'], ...], ...)`。此外，还需保证分词在 vocab 文件中存在。

- 问：如何实现 TinyBERT 和 FastBERT 二重蒸馏？

  答：`TinyBERTClassifier` 训练完成后使用 `.to_bert()` 提取子模型为 BERT 重新保存，而后使用 `FastBERTClassifier` 常规读取生成的 checkpoint 和配置文件即可。

- 问：我可以用这个框架做哪些有趣的事情？

  答：可以用 `GPT2LM` 来生成古诗/小说，可以使用 `TransformerMT` 搭建简单的聊天机器人，可以组合 `ELECTRALM` 和 `BERTLM` 进行文本纠错等等。

- 问：无意义的 warning 信息太多，该怎么剔除？

  答：这是 tensorflow 一直饱受诟病之处，我们也与你一同深受困扰。暂时没有有效的同时，又兼容各个 tf 版本的解决方案。

## 尾声

框架目前主要为我个人及团队所用，靠兴趣推动至今。如果能受到更多人，包括您的认可，我们会愿意投入更多精力进行丰富与完善，早日推出第一个正式版本。如果您喜欢，请点个 star 作为支持。如果有希望实现的 SOTA 算法，留下 issue，我们会酌情考虑，并安排时间为你编写。任何需求和建议都不尽欢迎。最后，感谢你读到这里。