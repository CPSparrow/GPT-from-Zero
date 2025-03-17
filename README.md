# 从0开始的**中文GPT-2**训练冒险😘️

## todo

| 接下来的计划                                        |
|:----------------------------------------------|
| 增加 total tokens cnt 功能                        |
| 深入了解各个数据集，准备构建更“稠密”的数据                        |
| 阅读论文、查阅资料，了解batch_size,epoch和learning_rate的设置 |

## 进度(Updates)

> 以往的记录已经被转移到 [`milestone`](./milestone.md) 文件中。

- 25.03.17：

  这两日的工作总结：

    - 进行了大约6h的训练(使用autodl的vGPU)，因为lr和loss都很怪所以提前停止检查代码。
    - `settings`:增加了路径验证的部分
    - `train`:

        1. 删去label smoothing(虽然实验结果不足以对比，但是怎么想也没有必要)
        2. 调整保存记录的代码
        3. 为“单机多卡”以及“从中断的地方继续训练”做了准备，不过都是没有验证的代码

## 数据来源(部分并不会考虑，并非最新)

- 传统文化: 还需清洗，懒得搞就不搞了
- **基础语料**:目前就用这个了，质量比较高的中文语料，网安平台下载的
- cci: 来源同上，还没下载来
- oscar: 来自mnbvc的crawler文件夹
- ~~[CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020)~~ 通过百度网盘提供，所以不想搞
- [liwu/MNBVC](https://huggingface.co/datasets/liwu/MNBVC) 超大规模数据集，oscar是common crawl的清洗，可以作为初步预训练的语料。
- [c4](https://hf-mirror.com/datasets/allenai/c4/tree/main/en) 可以作为英文训练
- [smoltalk](https://opencsg.com/datasets/OpenCSG/smoltalk_chinese/files/main/data) 作为sft语料

## 数据集预处理笔记

|   数据集   |  数据条数   |   文件大小 | 来源            |
|:-------:|:-------:|-------:|:--------------|
|  news   | 67_6028 | 2.10GB | 网安平台  基础语料    |
| crawler | 38_9713 | 0.23GB | mnbvc crawler |
|  zhihu  | 65_3692 | 1.40GB | mnbvc zhihu   |
|   en    | 35_6317 | 0.77GB | 上述的 c4        |

## 环境配置笔记

主要是为了上云和重置的时候方便查阅：

- cuda/nvcc >= 12.4

installation:

```shell
pip install torch torchvision torchaudio
pip install transformers datasets accelerate optimum
pip install pandas ninja packaging
```

validation before flash attention 2:

```shell
ninja --version
echo $?
```

随后手动安装nvidia/apex和flash attention 2:

~~apex:~~
> 目前的代码使用的是 `adamw_torch_fused` ，不需要安装apex

```shell
git clone https://github.com/NVIDIA/apex
cd apex
NVCC_APPEND_FLAGS="--threads 4" pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 4" ./
```

flash attention：

```shell
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

安装后清理缓存：
pip:

```shell
pip cache purge
```

conda:
运行前查看：

```shell
conda clean --dry-run --all
```

```shell
conda clean --tarballs
conda clean --packages
```
