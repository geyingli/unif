<p align="center">
    <br>
    	<img src="./docs/pics/logo.png" style="zoom:74%"/>
    <br>
<p>
<p align="center">
    <a>
        <img src="https://img.shields.io/badge/build-passing-brightgreen">
    </a>
    <a>
        <img src="https://img.shields.io/badge/version-v2.5.21-blue">
    </a>
    <a>
        <img src="https://img.shields.io/badge/tensorflow-1.x\2.x-yellow">
    </a>
    <a>
        <img src="https://img.shields.io/badge/license-Apache2.0-red">
    </a>
</p>


兼容 Tensorflow1.x/2.x 的高层封装 (Transformer/GPT-2/BERT/ALBERT/UniLM/XLNet/ELECTRA 等)，使用简单的代码完成语言模型/文本分类/文本生成/命名实体识别/机器阅读理解/机器翻译/序列标注/知识蒸馏任务。适用于 NLP 从业者。

### 特性

- 高效调用：三行代码完成训练及推理
- 高效运行：一行代码设置多进程/多 GPU 并行
- 品类丰富：支持 40+ 模型类
- 高分保证：提供对比学习、对抗式训练等多项训练技巧
- 可供部署：导出模型 PB 文件，供线上部署

### 安装

``` bash
git clone https://github.com/geyingli/unif
cd unif
python3 setup.py install --user
```

若需卸载，通过 `pip3 uninstall uf` 即可。

### 快速上手

``` python
import uf

# 建模
model = uf.BERTClassifier(config_file="./ref/bert_config.json", vocab_file="./ref/vocab.txt")

# 定义训练样本
X, y = ["久旱逢甘露", "他乡遇故知"], [1, 0]

# 训练
model.fit(X, y)

# 推理
print(model.predict(X))
```

## 模型列表

| 领域 | API 				| 说明                                            |
| :----------- | :----------- | :------------ |
| 语言模型|[`BERTLM`](./examples/tutorial/BERTLM.ipynb) | 结合 MLM 和 NSP 任务，随机采样自下文及其他文档 |
| |[`RoBERTaLM`](./examples/tutorial/RoBERTaLM.ipynb) 		| 仅 MLM 任务，采样至文档结束 |
| |[`ALBERTLM`](./examples/tutorial/ALBERTLM.ipynb) 		| 结合 MLM 和 SOP，随机采样自上下文及其他文档 |
| |[`ELECTRALM`](./examples/tutorial/ELECTRALM.ipynb) 		| 结合 MLM 和 RTD，生成器与判别器联合训练 |
| |[`VAELM`](./examples/tutorial/VAELM.ipynb) | 可生成语言文本负样本，也可提取向量用于聚类 |
| |[`GPT2LM`](./examples/tutorial/GPT2LM.ipynb) | 自回归式文本生成 | - |
| |[`UniLM`](./examples/tutorial/UniLM.ipynb) | 结合双向、单向及 Seq2Seq 建模的全能语言模型 |
| |[`UniLMPrompt`](./examples/tutorial/UniLMPrompt.ipynb) | 加入 prompt，进一步实现语言模型与下游任务的统一 |
|文本分类 / 单label|[`TextCNNClassifier`](./examples/tutorial/TextCNNClassifier.ipynb) 		| 小而快 |
|| [`RNNClassifier`](./examples/tutorial/RNNClassifier.ipynb) 		| 经典 RNN/LSTM/GRU |
|| [`BiRNNClassifier`](./examples/tutorial/BiRNNClassifier.ipynb) 		| 双向获取更优表征 |
|| [`BERTClassifier`](./examples/tutorial/BERTClassifier.ipynb) 		| - |
|| [`XLNetClassifier`](./examples/tutorial/XLNetClassifier.ipynb) 		| - |
|| [`ALBERTClassifier`](./examples/tutorial/ALBERTClassifier.ipynb) 		| - |
| |[`ELECTRAClassifier`](./examples/tutorial/ELECTRAClassifier.ipynb) 		| - |
| |[`WideDeepClassifier`](./examples/tutorial/WideDeepClassifier.ipynb) | 通过 Wide & Deep 架构融合句子级别特征 |
| |[`SemBERTClassifier`](./examples/tutorial/SemBERTClassifier.ipynb) 		| 通过 SemBERT 架构融合字级别的特征 |
| |[`UDAClassifier`](./examples/tutorial/UDAClassifier.ipynb) 		| 结合一致性学习的半监督学习算法 |
| |[`PerformerClassifier`](./examples/tutorial/PerformerClassifier.ipynb) 		| 基于正交随机向量快速计算 attention，实现加速 |
|文本分类 / 多label| [`BERTBinaryClassifier`](./examples/tutorial/BERTBinaryClassifier.ipynb) 		| -  |
|| [`XLNetBinaryClassifier`](./examples/tutorial/XLNetBinaryClassifier.ipynb) 		| - |
| |[`ALBERTBinaryClassifier`](./examples/tutorial/ALBERTBinaryClassifier.ipynb) 		| - |
| |[`ELECTRABinaryClassifier`](./examples/tutorial/ELECTRABinaryClassifier.ipynb) 		| - |
| 回归| [`BERTRegressor`](./examples/tutorial/BERTRegressor.ipynb) | - |
|| [`WideDeepRegressor`](./examples/tutorial/WideDeepRegressor.ipynb) | 通过 Wide & Deep 架构融合句子级别特征 |
|序列标注|[`BERTSeqClassifier`](./examples/tutorial/BERTSeqClassifier.ipynb) 		| - |
|| [`ALBERTSeqClassifier`](./examples/tutorial/ALBERTSeqClassifier.ipynb) 		| - |
|| [`ELECTRASeqClassifier`](./examples/tutorial/ELECTRASeqClassifier.ipynb) 		| - |
|| [`BERTSeqCrossClassifier`](./examples/tutorial/BERTSeqCrossClassifier.ipynb) 		| 序列标注与文本分类相结合的多任务学习 |
| 命名实体识别|[`BERTNER`](./examples/tutorial/BERTNER.ipynb) 		| -  |
|| [`BERTCRFNER`](./examples/tutorial/BERTCRFNER.ipynb) 		| 结合 CRF |
|| [`BERTCRFCascadeNER`](./examples/tutorial/BERTCRFCascadeNER.ipynb) | 实体识别与分类同时进行的级联架构 |
|机器阅读理解| [`BERTMRC`](./examples/tutorial/BERTMRC.ipynb) 		| -  |
| |[`ALBERTMRC`](./examples/tutorial/ALBERTMRC.ipynb) 		| - |
| |[`SANetMRC`](./examples/tutorial/SANetMRC.ipynb) 		| 引入 sentence attention |
| |[`BERTVerifierMRC`](./examples/tutorial/BERTVerifierMRC.ipynb) | 抽取 answer span 的同时判断可答性 |
| |[`RetroReaderMRC`](./examples/tutorial/RetroReaderMRC.ipynb) | 抽取 answer span 的同时判断可答性 |
| 机器翻译| [`TransformerMT`](./examples/tutorial/TransformerMT.ipynb) | 共享词表，标准 Seq2Seq 架构 | - |
| 模型蒸馏|[`TinyBERTClassifier`](./examples/tutorial/TinyBERTClassifier.ipynb) 		| 大幅压缩模型参数，提速十倍以上 |
|| [`TinyBERTBinaryClassifier`](./examples/tutorial/TinyBERTBinaryClassifier.ipynb)     | - |
|| [`FastBERTClassifier`](./examples/tutorial/FastBERTClassifier.ipynb) 		| 动态推理，易分样本提前离开模型 |
| 图像分类 / 单label | [`PNasNetClassifier`](./examples/tutorial/PNasNetClassifier.ipynb) 		| 基于 AutoML 搜索最佳网络结构 |


## 建模

实际上建模所需的参数不在少数，因模型而异。为了简便起见，大多数设置了默认值。了解每一项参数的含义是十分有必要的。参数的命名与原论文保持一致，如果遇到不明白的参数，除了看源代码外，可以前往原论文寻找答案。以 `BERTClassifier` 为例，包含以下参数：

```python
model = uf.BERTClassifier(
    config_file,                # json格式的配置文件，通常可以在预训练参数包里找到
    vocab_file,                 # 一行一个字/词的txt文件
    max_seq_length=128,         # 切词后的最大序列长度
    label_size=2,               # label取值数
    init_checkpoint=None,       # 预训练参数的路径或目录
    output_dir="./output",      # 输出文件导出目录
    gpu_ids="0,1,3,5",          # GPU代号 (为空代表不使用GPU; 如果使用的是Nvidia显卡，需要预先安装CUDA及cuDNN，而后可以通过`nvidia-smi`指令查看可用GPU代号)
    drop_pooler=False,          # 建模时是否跳过 pooler 层
    do_lower_case=True,         # 英文是否进行小写处理
    truncate_method="LIFO",     # 输入超出`max_seq_length`时的截断方式 (LIFO:尾词先弃, FIFO:首词先弃, longer-FO:长文本先弃)
)
```

模型使用完毕后，若需要清理内存，可以使用 `del model` 删除模型，或通过 `model.reset()` 对模型进行重置。

## 训练

同样，训练也包含一些可自行调节的参数，有些参数甚至十分关键：

``` python
  model.fit(
      X=X,                    # 输入列表
      y=y,                    # 输出列表
      sample_weight=None,     # 样本权重列表，放空则默认每条样本权重为1.0
      X_tokenized=None,       # 输入列表 (已预先分词处理的`X`)
      batch_size=32,          # 每训练一步使用多少数据
      learning_rate=5e-05,    # 学习率
      target_steps=None,      # 放空代表直接不间断地训练到`total_steps`；否则为训练停止的位置
      total_steps=-3,         # 模型计划训练的总步长，决定了学习率的变化曲线；正数，如1000000，代表训练一百万步；负数，如-3，代表根据数据量循环三轮的总步长
      warmup_ratio=0.1,       # 训练初期学习率从零开始，线性增长到`learning_rate`的步长范围；0.1代表在前10%的步数里逐渐增长
      print_per_secs=1,       # 多少秒打印一次训练信息
      save_per_steps=1000,    # 多少步保存一次模型参数
      **kwargs,               # 其他训练相关参数，如分层学习率等，下文将介绍
  )
```

### 断点

训练过程中，通常需要设立多个断点进行模型验证，决定是否停止训练。`target_steps` 正是为设置断点而存在的。以下是使用示例：

``` python
num_loops = 10      # 假设训练途中一共设置10个断点
num_epochs = 6      # 假设总共训练6轮

for loop_id in range(10):
    model.fit(
        X, y,
        target_steps=-((loop_id + 1) * num_epochs / num_loops),  # 训练断点 (-0.6, -1.2, ...)
        total_steps=-num_epochs,                                 # 训练全长 (-6)
    )
    print(model.score(X_dev, y_dev))                             # 验证模型
    model.localize(f"breakpoint.{loop_id}", into_file=".unif")   # 保存模型配置到`into_file` (同时保存模型参数到`output_dir`)
```

多次验证后表现最佳的断点，可以通过 `restore` 函数取用：
``` python
model = uf.restore("breakpoint.7", from_file=".unif")
```

从以上代码不难看出，`localize` 和 `restore` 函数是模型管理的利器。

### 多进程

`fit` 函数内部包含了两个步骤：

- 对输入进行预处理，转换为模型可接受的输入 (e.g. 整数/浮点数矩阵)

- 训练模型

当数据量变得庞大时，例如百万级，数据预处理可能要消耗十几二十分钟，这段期间 GPU 处于闲置状态，无疑是对资源的浪费，可以通过开启多进程处理加速这一过程 (注：不会对第二步模型训练加速)：

``` python
with uf.MultiProcess():
    X, y = ...            # 读取数据
    model.fit(X, y)
```

由于 python 中存在 PIL锁，每一个进程只能使用一个 CPU，那么多进程唤醒其他 CPU 的本质是对当前进程进行复制。因此需要注意的是，最好在大批量数据读到程序内存以前开启 `MultiProcess`，而不要在之后，否则每一个复制的进程都会拷贝一份完整数据，造成不必要的内存占用。

### TFRecords

当数据规模进一步增大，内存可能已经无法存放这样海量的数据，这时可以通过写入本地 TFRecords 文件，减小模型训练过程中的内存压力：

``` python
with uf.MultiProcess():
    X, y = ...            # 读取数据

    # 缓存数据
    model.to_tfrecords(
        X=X, y=y, sample_weight=None, X_tokenized=None,
        tfrecords_file="train.tfrecords",
    )

# 边读边训
model.fit_from_tfrecords(
    tfrecords_files=["train.tfrecords", ...],    # 支持同步从一个或多个TFRecords文件读取
    n_jobs=3,             # 启动三个线程
    batch_size=32,        # 以下参数和`fit`函数中参数相同
    learning_rate=5e-05,
    target_steps=None,
    total_steps=-3,
    warmup_ratio=0.1,
    print_per_secs=1,
    save_per_steps=1000,
    **kwargs,
)
```

实际上，也就是把 `fit` 函数中预处理和模型训练的两个步骤给分开。因此如果需要反复使用同一套数据进行训练，通过以上方式处理能节省更多时间。

### 预训练参数

预训练参数的 `ckpt` 文件中，每一个变量都有独立的命名和规格，如 `("layer_1/attention/self/query/kernel", [768, 768])`。在上文“模型列表”的详细链接中，我们列示了可以直接读取的公开预训练参数，从这些来源下载的预训练参数无需更多处理。但在其他地方获取的预训练参数，可能会存在与本框架中模型命名/规格不一致的情况。

规格不一致时，变量不可读取。但只有命名不一样时，可以通过下面的方法构建映射，将参数读到模型中：

```python
# 初始化模型，触发读取`ckpt`文件，查看哪些变量初始化失败
model.init()
print(model.uninited_vars)

# 人工进行变量名映射，并重新读取预训练参数
print(uf.list_variables(model.init_checkpoint))    # 在打印的结果中找到对应的参数名
model.assignment_map["layer_1/attention/self/query/kernel"] = model.uninited_vars["bert/encoder/layer_1/attention/self/query/kernel"]    # 添加映射关系
model.reinit_from_checkpoint()                     # 重新读取预训练参数
print(model.uninited_vars)                         # 在打印的结果中看看初始化失败的变量是否已消失
```

`ckpt` 是 tensorflow 输出的预训练参数，如果希望读取 PyTorch 输出的预训练参数，则稍微繁琐一些，可以通过将参数读到内存中，使用下面的变量赋值的方法实现。

### 变量赋值

将内存中的矩阵直接赋值给模型变量：

```python
import numpy as np

array = np.array([[0, 1, 2], [3, 4, 5]])  # 使用numpy.Array格式
print(model.global_variables)             # 查看所有变量
variable = model.global_variables[5]      # 获取变量
model.assign(variable, array)             # 赋值
print(model.sess.run(variable))           # 查看是否赋值成功
```

## 推理/评分

大多数模型的推理/评分只有以下几个参数，非常简单：

``` python
# 推理
model.predict(X=X, X_tokenized=None, batch_size=8)

# 评分
model.score(X=X, y=y, sample_weight=None, X_tokenized=None, batch_size=8)
```

与训练不同的是，推理/评分暂时不支持多进程加速和写入 TFRecords。如果需要推理海量数据，可以通过分批处理达成目的。

## Tricks

模型训练中包含的一些可改动的细节以及训练技巧，在 `fit` 和 `fit_from_tfrecords` 函数尾部添加参数即可实现。

### 优化器

自神经网络火热开始，逐渐演化出了一众优秀的最优化算法，包括 Gradient Descent (GD)、Momentum、Adaptive Gradient (AdaGrad)、Root Mean Square prop (RMSprop)、Adaptive Moment estimation (Adam) 等。时至今日，最常见的依然是 2018 年 BERT 所使用的 Adam Weight Decay Regularization (AdamW) 算法。但据实验表明，当训练的 `batch_size` 达到 512 以上时，AdamW 的收敛效率会极速下降，因而有了后来宣称能够支持大容量 batch 收敛，在 76 分钟内完成 BERT 预训练的 Layer-wise Adaptive Moments optimizer for Batching training (LAMB) 算法。这一套算法的表现基本优于 AdamW 或与之相当，推荐尝试。

```python
model.fit(..., optimizer="gd")
model.fit(..., optimizer="adam")
model.fit(..., optimizer="adamw")     # 默认adamw
model.fit(..., optimizer="lamb")
```

### 分层学习率

迁移学习中常见灾难性遗忘问题 (Catastrophic Forgetting)：模型急不可耐地适应新数据，而丢失了预训练中学到的知识。为了对抗这种过拟合，分层学习率是有效的 trick。越靠近输出层，学习率越大，反之亦然。启用方法是增加 `layerwise_lr_decay_ratio` 参数并设定一个 0 到 1 之间的浮点数。模型参数会通过 `.decay_power` 找到自己对应的指数，而后计算学习率：

$$r_t(w) = r_t^* \times LLDR^{f(w)}$$

$r^*$ 是当前训练阶段的全局学习率， $f$ 是参数到指数的映射。实际使用中可按需求调整参数的值：
```python
model.fit(..., layerwise_lr_decay_ratio=0.85)
print(model.decay_power)            # 可in-place修改
```

### 对抗式训练

在 2020 年前后，对抗式训练因其较好的效果开始从小众变成人尽皆知。与 CV 领域通过 GAN 的范式进行图像生成的对抗式训练不同，NLP 领域的对抗式训练指的是在梯度正方向上添加扰动 (而不是随机扰动)，以增强 embedding 泛化性和鲁棒性的策略。从最经典的 FGSM 开始，对抗式训练算法也在随时间推进，逐渐优化。在 2020 年，微软推出 SMART 算法成为该领域的 SOTA，在 GLUE 榜单榜上有名。但由于该算法只能在单 label 分类的场景使用，且需要调节的参数较多，因而也存在诸多不便之处。

```python
model.fit(..., adversarial="fgm", epsilon=0.5)                  # FGM
model.fit(..., adversarial="pgd", epsilon=0.05, n_loop=2)       # PGD
model.fit(..., adversarial="freelb", epsilon=0.3, n_loop=3)     # FreeLB
model.fit(..., adversarial="freeat", epsilon=0.001, n_loop=3)   # FreeAT
model.fit(..., adversarial="smart", epsilon=0.01, n_loop=2, prtb_lambda=0.5, breg_miu=0.2, tilda_beta=0.3)    # SMART (仅Classifier可用)
```

### R-Drop

全称 Regularized Dropout，同样是行之有效的在对抗过拟合的同时增强泛化能力的策略。策略思想极其简单，因而在大多数场景都能应用，在于将相同的数据经过不同的 dropout 后得到各自的概率分布，计算双向 KL 散度并加入到损失函数中。这一想法与对比学习、一致性学习颇为类似，但作用的变量是 dropout。

```python
model.fit(..., rdrop=True, alpha=1.0)     # alpha是损失项乘子
```

### InfoNCE Loss

凝聚了对比学习的核心思想 —— 在编码上推远负例、拉进正例。正例是相同样本在两次前馈中不同 dropout 下的结果，负例是同一 batch 下的其他样本。私以为，对比学习作为当前 sentence embedding 领域的 SOTA，值得作为多任务学习的 trick，加入到有监督的训练任务中。这里我们实现了 SimCSE 原论文中无监督的训练方案：

$$L_i=-log\frac{e^{sim(h_i,h_i')/\tau}}{\sum_j e^{sim(h_i,h_j')/\tau}}$$

$\tau$ 为温度系数，即为下面函数中的 `tau`。

```py
model.fit(..., info_nce_loss=True, tau=1.0, alpha=0.05)     # alpha是损失项乘子
```

### Focal Loss

在基于交叉熵损失的分类场景下，动态调节易/难分样本的损失大小，从而使训练将更多的注意力放在难分样本上。是解决类型不平衡问题的绝佳手段，原论文引用次数已超过 1.5w。

$$L_i(p)=-\alpha(1-p)^{\gamma} log(p)$$

$\gamma$ 即为下述函数中的参数 `gamma`，可取任意大于 0 的值； $\alpha$ 代表类别权重，同一 label 下的 $\alpha$ 是一致的。由于 $\alpha$ 与本框架的 `sample_weight` 理念重合，使用参数 `sample_weight` 即可达到调节 $\alpha$ 的目的。

```python
X = ["孔雀东南飞", "五里一徘徊", "不知天上宫阙", "今夕是何年"]
y = [2, 0, 1, 1]
alpha_map = {0: 1.8, 1: 1.0, 2: 1.5}
sample_weight = [alpha_map[_y] for _y in y]     # 样本权重，上文中有介绍
model.fit(X, y, sample_weight, focal_loss=True, gamma=1.0)
```


## TFServing

导出供部署上线的 PB 文件到指定目录下：

``` python
model.export(
    "tf_serving/model",     # 导出目录
    rename_inputs={},       # 重命名输入
    rename_outputs={},      # 重命名输出
    ignore_inputs=[],       # 剪裁输入
    ignore_outputs=[],      # 裁剪输出
)
```

而后的模型服务化步骤在 cpp、go、java 等多种语言上都能实现，涉及到的后端代码以及部署上线已与本框架无关，这里就不再展示了。

## 开发需知

核心的代码架构如下图所示。新的模型类需要在 apps 目录下添加，而建模则在 model 目录下。欢迎一切有效的 pull request。

<p align="center">
    <br>
    	  <img src="./docs/pics/framework.png" style="zoom:50%"/>
    <br>
<p>

## FAQ

- 问：有什么提高训练速度的方法吗？

  答：首先是几种能立即实施的基础方法：减小 max_seq_length，多 GPU 并行，多进程数据处理，以及梯度累积。在这些之外，可以进一步尝试对输入的数据进行拆分，在训练过程中逐步提高 max_seq_length、batch_size 和 dropout_rate (通过提高拟合速度，缩短整个训练周期)。当然，还有一些在 UNIF 暂时无法实现的功能，可以前往其他 repo 寻求解决方案，包括但不限于混合精度训练、OP融合、使用 Linformer 等时间复杂度小于 $O(N^2)$ 的模型。

- 问：训练时内存不足，该怎么办？

  答：首先需要明确是显存溢出还是内存爆炸。如果是显存溢出，则需要降低 batch_size；如果是由于数据体量过于庞大导致的内存爆炸，可以尝试通过 `model.to_tfrecords()` 分批将数据写入 TFRecords 文件，而后清出内存，通过 `model.fit_from_tfrecords()` 读取进行训练。

- 问：模型输入有什么限制吗？

  答：对于大多数模型来说，没有限制。一条样本，可以是一个字符串，也可以是多个字符串。以 BERT 为例，完全不必局限于一到两个句子的输入，而是可以通过 list 组合多个 segment，e.g.  `X = [["文档1句子1", "文档1句子2", "文档1句子3"], ["文档2句子1", "文档2句子2"]]`，模型会自动按顺序拼接并添加分隔符。

- 问：如何查看切词结果？

  答：通过 `model.tokenizer.tokenize(text)` 可查看切词结果。另外也可通过 `model.convert(X)` 查看经过预处理的实际的模型输入。

- 问：如果我想使用自己的切词工具，该怎么做？

  答：提前使用自己的工具做好分词，而后在训练和推理时将原先的传入参数由 `X` 改为 `X_tokenized`。例如，原本传入 `model.fit(X=["黎明与夕阳", ...], ...)`，使用自己的分词工具后，改为传入 `model.fit(X_tokenized=[["黎", "##明", "与", "夕", "##阳"], ...], ...)`。此外，还需保证分词在 vocab 文件中存在。

- 问：无意义的 warning 信息太多，该怎么剔除？

  答：这是 tensorflow 一直饱受诟病之处，我也与你一同深受困扰。可以试着在运行时屏蔽 WARNING 类型的信息：`python3 train.py | grep "WARNING" -v`。

## Tips

- 经典永不衰：

  模型并非越复杂表现越好，在文本生成以外的应用上，BERT 几乎足够让你一招鲜吃遍天。

- 推荐的预训练参数：

  优先使用哈工大训的 `macbert-base`，亲测各项任务上表现都很不错。

- TinyBERT 搭配 FastBERT 进行二重蒸馏：

  `TinyBERTClassifier` 训练完成后使用 `.to_bert()` 提取子模型为 BERT，而后使用 `FastBERTClassifier` 读取，继续进行提速。

- 实现一些有趣的事情：

  用 `GPT2LM` 来生成古诗/小说，用 `TransformerMT` 搭建简单的聊天机器人，或组合 `ELECTRALM` 和 `BERTLM` 进行文本纠错等等。


## 尾声

ChatGPT 的出现让 NLP 领域变得妙不可言。我已退出原就职公司，亦退出这个行业，此仓库因此搁置。虽然只有寥寥的一百多人关注，但站在我面前也是不小的队伍，真诚感谢大家的关注。有问题依然可以提问在 issue 区，我会尽快回答。
