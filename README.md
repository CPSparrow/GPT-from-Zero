# 从0开始的**中文GPT-2**训练冒险😘️

## 进度(Updates)

- 25.03.15:

  论文汇报暂时告一段落，多出来的时间应该会拿来重构这个项目从而准备下一次训练。

  重要的参考资料：[知乎 - Training Arguments](https://zhuanlan.zhihu.com/p/662619853)

  计划是分为三步走：
    - 整理、阅读原先代码
    - 重写dataloader以及相关部分，中间可能会涉及 `dataclass` 的学习。
    - 可能会使用手动设计的代码训练，验证和保存模型。
- 25.03.01:\
  使用平台的vGPU开始训练。总共约 `7.7GB` 的数据，230M+ (?) 参数的模型,设计了 learning_rate = 2.5e-4, weight_decay =
  0.0001。完整训练完估计用时超过300小时，肯定不会把100个 epoch 都跑完。\
  > 事后笔记：并非7GB，实际上要少的多

  开始训练之后许久才发现一个很好的教程，来自 huggingface
  的[NLP Course](https://huggingface.co/learn/nlp-course/zh-CN/chapter7/6?fw=pt)。对比之下感觉这一次的训练结果不会乐观。\
  几个主要的差异：
    - 没有手动设置 warmup 和 lr_scheduler
    - course中居然只用了一个epoch效果就很好。难以置信。
    - 对于示例的小模型(124M), lr 更高，达到 5e-4 ; weight_decay 也更高，达到 0.1 (而我根据知乎上某篇文章选择了 1e-4 )
    - 也是我欠缺考虑，设置了过大的 max_length: 1024 tokens。教程中的 128 tokens 其实是一个很好的例子：开头不应该太贪心。

  根据目前的设置再训练一会儿(也许一天，也许多一点)，看看结果再说吧。
- 25.02.28:\
  使用L20训练了4个小时多一点的时间，大概是16_0000条数据(?)
  两个epoch。因为数据的改动主要在服务器所以细节有些遗忘。结论是效果果然很差，只会复读无意义的符号。因此重新速读了GPT1、2的论文，对比下主要有以下几个不同，可能是训练效果差的原因：
    - 学习率太低。原先的设置遵循了hugginface的设置，但是2e-5是按照微调的情况设置的
    - 训练太少。gpt1中似乎提到总共训练了100个epoch，而gpt2中又提到原始的数据(如果算上后来被清除的WikiText)
      则有足足40GB。假如说是1.5B -> 40GB,那么等比例来看本次研究的0.3B模型应当是8GB。
      不过这一方面还不是很确定，可能还得再看一些资料来决定。
    - 基于Amper(L20)的修改还没有同步，本次提交的主要是整理bpe的代码以及本地的重新训练的分词器(分别是7个epoch的
      ```32k_v1```和14个epoch的```32k_v2```),比较之后也是确定了目前来说似乎v1便已经足够合适。
- 25.02.26:\
  用8_0000行(约228MB)中文训练了两个epoch，但是模型完全无法输出。\
  举例：\
  输入：村民们都很高兴\
  输出：村民们都很高兴，在“““””””””””””””””””””””””””””””””““““““““““““““““\
  原因猜测：
    - vocab_size,embed_dim或是train_steps太小导致的训练不足(训练时eval loss > 7)
    - 补充：也有可能是weight_decay的原因。至于其他的超参数则不清楚了。
- 25.02.25:\
  重写了BPE的训练。基于AutoTokenizer的训练使得后续可以方便的调用。同时基本写好了训练的框架(即```prepare.py```文件)
  。稍后的任务是调整正确的DataLoader和DataCollector
- 25.02.24:\
  周末休息之后重新查阅资料，准备重写代码。对于接下来的基本计划如下
    - 借助AutoTokenizers重新训练BPE(而且似乎这样可以直接得到FastTokenizer),需要适当调整special tokens
    - 查询类似的方法初始化GPT2模型并训练

    - 目前不是很确定的部分: Padding Strategy 的处理和截断的处理

- 25.02.20:\
  bpe代码基本完成，随后发现训练bpe只需要大约1GB的数据即可，而非原先预想的要把所有数据跑一遍(喜)
- 25.02.18～29:\
  下载数据集，熟悉读取代码

## 数据来源(部分不考虑)

- 传统文化: 还需清洗，懒得搞就不搞了
- **基础语料**:目前就用这个了，质量比较高的中文语料，网安平台下载的
- cci: 来源同上，还没下载来
- oscar: 来自mnbvc的crawler文件夹
- ~~[CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020)~~ 通过百度网盘提供，所以不想搞
- [liwu/MNBVC](https://huggingface.co/datasets/liwu/MNBVC) 超大规模数据集，oscar是common crawl的清洗，可以作为初步预训练的语料。
- [c4](https://hf-mirror.com/datasets/allenai/c4/tree/main/en) 可以作为英文训练
- [smoltalk](https://opencsg.com/datasets/OpenCSG/smoltalk_chinese/files/main/data) 作为sft语料

## 规划(最初的设想，并非最新)

最开始的规模应该向gpt-2看齐(也许吧)。

0. 训练tokenizer -> BPE ~~和embedding(?)~~
    - 已完成: 测试```load_jsonl```函数
    - 已完成(略微存疑？): BBPE
1. 基础Transfomer
    - 使用torch，或者transformers?
2. 优化的transformer:RoPE,flash attention,swi激活函数，也许还有其他
    - 多卡训练
    - 更大模型
    - accelerate库
3. MoE架构，DS MoE？
