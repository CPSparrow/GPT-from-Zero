# 从0开始的**中文GPT-2**训练冒险😘️

## todo & notes

不错的参考资料：[GPT-2复现笔记](https://zhuanlan.zhihu.com/p/16880416388)

| 接下来的计划 |
|:-------|
| NaN    |

## 进度(Updates)

> 以往的记录已经被转移到 [`milestone`](./milestone.md) 文件中。

- 25.03.29:

  > 补充原先的训练结果：大约基于 0.5B tokens ，2 epoch 训练的 GPT-2 like 模型(300M)。模型没有体现出对语言的理解能力，只是复读和重复最开始的输出。

  接下来的计划 ： **再次重构该项目**

  主要的变化将会是：

    - 重新下载更大的数据
    - 使用语料库从0训练 tokenizer ,预计训练3个不同的版本。
    - 使用 LLaMa 的架构训练小模型。超参数可能参考上述 **复现笔记** ，或是之后去阅读LLaMa的论文。

- 25.03.17：

  这两日的工作总结：

    - 进行了大约6h的训练(使用autodl的vGPU)，因为lr和loss都很怪所以提前停止检查代码。
    - `settings`:增加了路径验证的部分
    - `train`:

        1. 删去label smoothing(虽然实验结果不足以对比，但是怎么想也没有必要)
        2. 调整保存记录的代码
        3. 为“单机多卡”以及“从中断的地方继续训练”做了准备，不过都是没有验证的代码

## 数据来源(未更新)

- 传统文化: 还需清洗，懒得搞就不搞了
- **基础语料**:目前就用这个了，质量比较高的中文语料，网安平台下载的
- cci: 来源同上，还没下载来
- oscar: 来自mnbvc的crawler文件夹
- ~~[CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020)~~ 通过百度网盘提供，所以不想搞
- [liwu/MNBVC](https://huggingface.co/datasets/liwu/MNBVC) 超大规模数据集，oscar是common crawl的清洗，可以作为初步预训练的语料。
- [c4](https://hf-mirror.com/datasets/allenai/c4/tree/main/en) 可以作为英文训练
- [smoltalk](https://opencsg.com/datasets/OpenCSG/smoltalk_chinese/files/main/data) 作为sft语料

## 环境配置笔记

主要是为了上云和重置的时候方便查阅：

- cuda/nvcc >= 12.4

installation:

```shell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
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
