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


def format_on_batch(dict_samples):
    result = list()
    batched_samples = dict_samples['段落']
    for sample in batched_samples:
        assert isinstance(sample, list)
        result.append("".join([i['内容'] for i in sample]))
    return {'Content': result}


def data_format():
    n_proc = 12
    batch_size = 5_000
    
    if not os.path.exists(os.path.join(cleaned_file_dir, "wiki")):
        wiki = load_dataset(
            path=os.path.join(raw_file_dir, "wiki"),
            cache_dir=cache_dir,
            split="train[:]",
            num_proc=n_proc,
        )
        wiki = wiki.map(
            function=format_on_batch,
            num_proc=n_proc,
            batched=True,
            batch_size=batch_size,
            remove_columns=wiki.column_names,
            desc="处理 Wiki 数据",
        )
        wiki.save_to_disk(os.path.join(cleaned_file_dir, "wiki"))
    else:
        wiki = load_from_disk(os.path.join(cleaned_file_dir, "wiki"))
    
    if not os.path.exists(os.path.join(cleaned_file_dir, "crawl_oscar")):
        crawl_oscar = load_dataset(
            path=os.path.join(raw_file_dir, "crawl_oscar"),
            cache_dir=cache_dir,
            split="train[:]",
            num_proc=n_proc,
        )
        crawl_oscar = crawl_oscar.map(
            function=format_on_batch,
            num_proc=n_proc,
            batched=True,
            batch_size=batch_size,
            remove_columns=crawl_oscar.column_names,
            desc="处理 Oscar 数据",
        )
        crawl_oscar.save_to_disk(os.path.join(cleaned_file_dir, "crawl_oscar"))
    else:
        crawl_oscar = load_from_disk(os.path.join(cleaned_file_dir, "crawl_oscar"))
    
    return wiki, crawl_oscar


if __name__ == "__main__":
    wiki, oscar = data_format()
    print(wiki)
    print(oscar)
