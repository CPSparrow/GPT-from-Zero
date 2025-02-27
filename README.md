# 从0开始的llm训练冒险(并非)

## 进度(Updates)

- 25.02.26: 用8_0000行(约228MB)中文训练了两个epoch，但是模型完全无法输出。\
举例：\
```[{'generated_text': '村民们都很高兴，在“““””””””””””””””””””””””””””””””““““““““““““““““'}]```\
原因猜测：vocab_size,embed_dim或是train_steps太小导致的训练不足(训练时eval loss > 7)
    - 补充：也有可能是weight_decay的原因。至于其他的超参数则不清楚了。
- 25.02.25: 重写了BPE的训练。基于AutoTokenizer的训练使得后续可以方便的调用。同时基本写好了训练的框架(即```prepare.py```文件)。稍后的任务是调整正确的DataLoader和DataCollector
- 25.02.24: 周末休息之后重新查阅资料，准备重写代码。对于接下来的基本计划如下
    - 借助AutoTokenizers重新训练BPE(而且似乎这样可以直接得到FastTokenizer),需要适当调整special tokens
    - 查询类似的方法初始化GPT2模型并训练

    - 目前不是很确定的部分: Padding Strategy 的处理和截断的处理

- 25.02.20: bpe代码基本完成，随后发现训练bpe只需要大约1GB的数据即可，而非原先预想的要把所有数据跑一遍(喜)
- 25.02.18～29: 下载数据集，熟悉读取代码

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