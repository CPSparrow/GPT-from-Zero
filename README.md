# 从0开始的llm训练冒险(并非)

## 规划

最开始的规模应该向gpt-2看齐(也许吧)。

0. 训练tokenizer -> BPE ~~和embedding(?)~~
    - 已完成: 测试```load_jsonl```函数 
    - 已完成(略微存疑？): BBPE
1. 基础Transfomer
    - 使用torch，或者transformers?
2. 优化的transformer：RoPE,flash attention,swi激活函数，也许还有其他
    - 多卡训练
    - 更大模型
    - accelerate库
3. MoE架构，DS MoE？

## 进度

- 25.02.20： bpe代码基本完成，随后发现训练bpe只需要大约1GB的数据即可，而非原先预想的要把所有数据跑一遍(喜)
- 25.02.18～29： 下载数据集，熟悉读取代码

## 数据来源(部分不考虑)
- 传统文化： 还需清洗，懒得搞就不搞了
- **基础语料**：目前就用这个了，质量比较高的中文语料，网安平台下载的
- cci: 来源同上，还没下载来
- oscar： 来自mnbvc的crawler文件夹
- ~~[CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020)~~ 通过百度网盘提供，所以不想搞
- [liwu/MNBVC](https://huggingface.co/datasets/liwu/MNBVC) 超大规模数据集，oscar是common crawl的清洗，可以作为初步预训练的语料。
- [c4](https://hf-mirror.com/datasets/allenai/c4/tree/main/en) 可以作为英文训练
- [smoltalk](https://opencsg.com/datasets/OpenCSG/smoltalk_chinese/files/main/data) 作为sft语料