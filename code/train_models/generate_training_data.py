import os
from settings import config
from transformers import AutoTokenizer
from datasets import (
    load_dataset,
    concatenate_datasets,
    load_from_disk,
    Dataset,
)

data_dir = config.data_dir
cache_dir = os.path.join(data_dir, "cache")
raw_file_dir = os.path.join(data_dir, "raw")
cleaned_file_dir = os.path.join(data_dir, "cleaned")
tokenized_file_dir = os.path.join(data_dir, "tokenized")


def extract_from_batch(dict_samples):
    """
    Core function for extracting data from batch,
    suitable for some data from MNBVC
    """
    result = list()
    batched_samples = dict_samples['段落']
    for sample in batched_samples:
        assert isinstance(sample, list)
        result.append("".join([i['内容'] for i in sample]))
    return {'Content': result}


def get_lm_data():
    """
    Return the tokenized data.
    """
    
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
    
    max_length = config.max_length
    
    if os.path.exists(tokenized_file_dir):
        tokenized_data = load_from_disk(tokenized_file_dir)
    else:
        tokenized_data = load_from_disk(os.path.join(cleaned_file_dir, "crawl_oscar"))
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(
                config.bpe_dir, '32k_v2_1'
            )
        )
        tokenized_data = tokenized_data.map(
            tokenize, batched=True, num_proc=20,
            remove_columns=tokenized_data.column_names,
            desc="running tokenizer",
        )
        print("splitting dataset")
        tokenized_data = tokenized_data.train_test_split(test_size=0.05, shuffle=False)
        tokenized_data.save_to_disk(tokenized_file_dir)
    return tokenized_data


def extract_content():
    """
    Clean the Dataset to be a single column named 'Content',
    will return a dataset object
    """
    n_proc = 12
    batch_size = 5_000
    
    if not os.path.exists(os.path.join(cleaned_file_dir, "crawl_oscar")):
        crawl_oscar = load_dataset(
            path=os.path.join(raw_file_dir, "crawl_oscar"),
            cache_dir=cache_dir,
            split="train[:]",
            num_proc=n_proc,
        )
        crawl_oscar = crawl_oscar.map(
            function=extract_from_batch,
            num_proc=n_proc,
            batched=True,
            batch_size=batch_size,
            remove_columns=crawl_oscar.column_names,
            desc="处理中文爬虫数据",
        )
        crawl_oscar = crawl_oscar.shuffle(seed=42, writer_batch_size=3000)
        crawl_oscar.save_to_disk(os.path.join(cleaned_file_dir, "crawl_oscar"))
    else:
        crawl_oscar = load_from_disk(os.path.join(cleaned_file_dir, "crawl_oscar"))
    
    return crawl_oscar


if __name__ == "__main__":
    # oscar = extract_content()
    # print(oscar)
    tokenized_data = get_lm_data()
    # 1.77e+09 tokens
    print(
        f"{len(tokenized_data['train'])} samples,{len(tokenized_data['train']) * config.max_length:.2e} tokens for train split")
