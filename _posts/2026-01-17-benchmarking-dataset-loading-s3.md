---
layout: post
title: Benchmarking Dataset Loading from S3 - Datasets vs LitData vs DeepLake
date: 2026-01-17 1:00:00
description: A comparison of loading datasets from S3 using HuggingFace Datasets, Lightning Data (LitData), and DeepLake.
tags: benchmarking python datasets s3 deep-learning
categories: blog
---

Loading large datasets efficiently from cloud storage like S3 is a critical step in many machine learning pipelines. In this post, I compare three options: **HuggingFace Datasets**, **Lightning Data (LitData)**, and **DeepLake**.

I used the [SUN397](https://huggingface.co/datasets/tanganke/sun397) dataset for these benchmarks. The goal was to compare the ease of use, data preparation speed, and loading performance directly from S3.

## Comparison & Results

### Data Preparation Speed

To use **LitData** and **DeepLake**, you must first transform the dataset into their optimized formats (sharded files, tensor formats, etc.) to enable efficient streaming and random access. **HuggingFace Datasets** can also be saved to disk / S3 using its native serialization.

In my experiments, the data preparation process for **LitData** was significantly faster than **DeepLake**. LitData's optimization process is parallelized and seemingly more efficient for this image dataset.

### Loading from S3 & Streaming

The main difference lies in how these libraries handle S3 data at runtime.

1.  **HuggingFace Datasets (`load_from_disk`)**: When loading a dataset saved to S3, it **downloads the entire dataset** to a local cache before it becomes usable. This is a major bottleneck for large datasets if you don't want to duplicate the data locally.
2.  **LitData & DeepLake**: Both support **streaming** directly from S3. They fetch data chunks as needed during iteration without a full pre-download.

In terms of streaming performance, **LitData** proved to be way faster than **DeepLake** during iteration in my tests.

## Conclusion

For scenarios where you need to stream datasets from S3 without a full download:

1.  **HuggingFace Datasets** is less suitable because `load_from_disk` enforces a full download/cache step.
2.  **LitData** is the superior choice in this benchmark, outperforming **DeepLake** in both preparation speed and streaming throughput.

If you are looking for a performant solution to host and stream your datasets from self-hosted S3 buckets, **LitData** appears to be the superior choice based on these benchmarks.

## Implementation Details

Below is the code used for the benchmarks.

### Data Preparation Code

**HuggingFace Datasets**
The standard `save_to_disk` approach simply serializes the dataset.

```python
import datasets
import os

ds = datasets.load_dataset("tanganke/sun397", split="train")
ds.save_to_disk(os.path.expanduser("~/sun397-hf"))
```

**DeepLake**
DeepLake requires defining a schema (columns) and iterating through the dataset to append items.

```python
import datasets
import deeplake
import numpy as np
from tqdm import tqdm
import os

ds = datasets.load_dataset("tanganke/sun397", split="train")
deeplake_ds = deeplake.create(os.path.expanduser("~/sun397-train"))

deeplake_ds.add_column("image", deeplake.types.Image())
deeplake_ds.add_column("label", deeplake.types.ClassLabel(dtype="Int64"))

for item in tqdm(ds):
    deeplake_ds.append(
        [
            {
                "image": np.array(item["image"].convert("RGB")),
                "label": item["label"],
            }
        ]
    )
```

**LitData**
LitData uses an `optimize` function that parallelizes the processing.

```python
import datasets
import litdata as ld
import os

ds = datasets.load_dataset("tanganke/sun397", split="train")

def get_item(idx):
    item = ds[idx]
    image = item["image"]
    label = item["label"]
    return {"image": image, "label": label}

if __name__ == "__main__":
    ld.optimize(
        fn=get_item,
        inputs=list(range(len(ds))),
        output_dir=os.path.expanduser("~/sun397-litdata"),
        num_workers=0,
        chunk_bytes="64MB",
    )
```

### Loading from S3 Code

**HuggingFace Datasets**

```python
import fsspec
from datasets import load_from_disk
from tqdm import tqdm

# This will download the dataset from S3 before loading (create local cache)
ds = load_from_disk(
    "s3://ml-data/datasets-disk/sun397-train", keep_in_memory=False
)
print(len(ds))
for item in tqdm(ds):
    pass
```

**DeepLake**

```python
import os
import deeplake
from tqdm import tqdm

ds = deeplake.open_read_only(
    "s3://ml-data/deeplake-datasets/sun397-train"
).pytorch()
print(len(ds))
for item in tqdm(ds):
    pass
```

**LitData**

```python
import litdata as ld
from tqdm import tqdm

ds = ld.StreamingDataset(
    "s3://ml-data/litdata-datasets/sun397-train",
)
print(len(ds))
for item in tqdm(ds):
    pass
```
