import os
import sys
from datasets import load_dataset,interleave_datasets,load_from_disk


def get_data():
    base_pwd = "/home/handsome-coder/桌面/corpus/1.pretrain"
    cache_dir = os.path.join(base_pwd, "cache")
    file_dir = os.path.join(base_pwd, "combined_data")
    if os.path.exists(file_dir):
        data=load_from_disk(file_dir)
    else:
        news = load_dataset(
            path=os.path.join(base_pwd, "cn_news"), cache_dir=cache_dir,
            split=f"train[:]", num_proc=12
        )
        news = news.map(
            lambda x: x, num_proc=12, batched=True, remove_columns=["ID"],
            desc="处理中文新闻"
        )
        
        crawler = load_dataset(
            path=os.path.join(base_pwd, "cn_common_crawl"), cache_dir=cache_dir,
            split=f"train[:]", num_proc=12
        )
        crawler = crawler.map(
            lambda x: {"Content": x["段落"][0]["内容"]}, num_proc=12,
            remove_columns=crawler.column_names, desc="处理中文爬虫语料"
        )
        
        zhihu = load_dataset(
            path=os.path.join(base_pwd, "cn_qa_zhihu"), cache_dir=cache_dir,
            split=f"train[:]", num_proc=12
        )
        zhihu = zhihu.map(
            lambda x: {"Content": "问：" + x["问"] + "答：" + x["答"]}, num_proc=12,
            remove_columns=zhihu.column_names, desc="处理知乎问答"
        )
        
        en = load_dataset(
            path=os.path.join(base_pwd, "en"), cache_dir=cache_dir,
            split=f"train[:]", num_proc=12
        )
        en = en.map(
            lambda x: {"Content": x["text"]}, num_proc=12,
            remove_columns=en.column_names, desc="处理英文语料"
        )
        
        data = interleave_datasets([news, crawler, zhihu, en])
        del news, crawler, zhihu, en
        data = data.train_test_split(test_size=0.2)
        data.save_to_disk(file_dir)
    return data

if __name__ == "__main__":
    data=get_data()
    print(data)
