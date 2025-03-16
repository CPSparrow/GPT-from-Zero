import os
import settings
from datasets import (
    load_dataset,
    interleave_datasets,
    load_from_disk,
    DatasetInfo,
    Dataset,
)
from transformers import AutoTokenizer

data_pwd = settings.config.data_pwd


def get_data():
    cache_dir = os.path.join(data_pwd, "cache")
    file_dir = os.path.join(data_pwd, "combined_data")
    
    if os.path.exists(file_dir):
        data = load_from_disk(file_dir)
    else:
        news = load_dataset(
            path=os.path.join(data_pwd, "cn_news"), cache_dir=cache_dir,
            split=f"train[:]", num_proc=12
        )
        news = news.map(
            lambda x: x, num_proc=12, batched=True, remove_columns=["ID"],
            desc="处理中文新闻"
        )
        
        crawler = load_dataset(
            path=os.path.join(data_pwd, "cn_common_crawl"), cache_dir=cache_dir,
            split=f"train[:]", num_proc=12
        )
        crawler = crawler.map(
            lambda x: {"Content": x["段落"][0]["内容"]}, num_proc=12,
            remove_columns=crawler.column_names, desc="处理中文爬虫语料"
        )
        
        zhihu = load_dataset(
            path=os.path.join(data_pwd, "cn_qa_zhihu"), cache_dir=cache_dir,
            split=f"train[:]", num_proc=12
        )
        zhihu = zhihu.map(
            lambda x: {"Content": "问：" + x["问"] + "答：" + x["答"]}, num_proc=12,
            remove_columns=zhihu.column_names, desc="处理知乎问答"
        )
        
        en = load_dataset(
            path=os.path.join(data_pwd, "en"), cache_dir=cache_dir,
            split=f"train[:]", num_proc=12
        )
        en = en.map(
            lambda x: {"Content": x["text"]}, num_proc=12,
            remove_columns=en.column_names, desc="处理英文语料"
        )
        
        data = interleave_datasets(
            datasets=[news, crawler, zhihu, en],
            probabilities=[0.33, 0.23, 0.33, 0.11],
            info=DatasetInfo(description="mixed dataset,to be tokenized.")
        )
        
        del news, crawler, zhihu, en
        data = data.train_test_split(test_size=0.01, shuffle=True)
        data.save_to_disk(file_dir)
    return data


def get_tokenized(data: Dataset):
    def preprocess(samples):
        return tokenizer(
            samples["Content"], return_tensors="pt",
            max_length=settings.config.max_length,
            truncation=True, padding=True,
        )
    
    tokenized_dir = os.path.join(data_pwd, "tokenized_data")
    
    if os.path.exists(tokenized_dir):
        data = load_from_disk(tokenized_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(
                settings.config.bpe_pwd, '32k_v1'
            )
        )
        data = data.map(
            preprocess, batched=True, num_proc=12,
            drop_last_batch=True,
            remove_columns=data["train"].column_names,
            desc="running tokenizer"
        )
        data.save_to_disk(tokenized_dir)
    return data


if __name__ == "__main__":
    data = get_data()
    data = get_tokenized(data)
    train = data["train"].to_iterable_dataset()
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=os.path.join(settings.config.bpe_pwd, '32k_v1')
    )
    cnt = 0
    for index, item in enumerate(train):
        if index >= 5000:
            break
        cnt += sum(item["attention_mask"])
    print(f"avg:{cnt / 5000:.2f}")
