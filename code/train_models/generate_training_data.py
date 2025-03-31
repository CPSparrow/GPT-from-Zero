import os
import settings
from datasets import (
    load_dataset,
    concatenate_datasets,
    load_from_disk,
    Dataset,
)

data_dir = settings.config.data_dir
cache_dir = os.path.join(data_dir, "cache")
raw_file_dir = os.path.join(data_dir, "raw")
cleaned_file_dir = os.path.join(data_dir, "cleaned")


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
    oscar = extract_content()
    print(oscar)
