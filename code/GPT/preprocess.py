import os
import settings
from datasets import (
    load_dataset,
    concatenate_datasets,
    load_from_disk,
    DatasetInfo,
    Dataset,
)
from transformers import AutoTokenizer

data_pwd = settings.config.data_pwd


def mix_raw_text():
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
            lambda x: x, num_proc=12, batched=True,
            remove_columns=["ID"], desc="处理中文新闻"
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
            lambda x: {"Content": "问：" + x["问"] + "答：" + x["答"]},
            num_proc=12, remove_columns=zhihu.column_names, desc="处理知乎问答"
        )
        
        en = load_dataset(
            path=os.path.join(data_pwd, "en"), cache_dir=cache_dir,
            split=f"train[:]", num_proc=12
        )
        en = en.map(
            lambda x: {"Content": x["text"]}, num_proc=12,
            remove_columns=en.column_names, desc="处理英文语料"
        )
        
        data = concatenate_datasets(
            dsets=[news, crawler, zhihu, en],
            info=DatasetInfo(description="concating dataset,to be tokenized.")
        )
        
        del news, crawler, zhihu, en
        data.save_to_disk(file_dir)
    return data


def get_tokenized():
    def tokenize(samples):
        outputs = tokenizer(
            samples["Content"],
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_length=True, stride=10,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == max_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}
    
    max_length = settings.config.max_length
    tokenized_dir = os.path.join(data_pwd, "tokenized_data")
    
    if os.path.exists(tokenized_dir):
        data = load_from_disk(tokenized_dir)
    else:
        data = load_from_disk(os.path.join(data_pwd, "combined_data"))
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(
                settings.config.bpe_pwd, '32k_v1'
            )
        )
        data = data.map(
            tokenize, batched=True, num_proc=12,
            remove_columns=data.column_names,
            desc="running tokenizer",
        )
        print("splitting dataset")
        data = data.train_test_split(test_size=5.5e-3, shuffle=True)
        data.save_to_disk(tokenized_dir)
    return data


if __name__ == "__main__":
    data = mix_raw_text()
    data = get_tokenized()
    print("===== data info =====")
    print(data)
