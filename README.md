# DualGeo

This is the code repository for the paper "DualGeo: A Dual-View Framework for Worldwide Image Geo-localization".

conference in **ICME 2026.**




## Abstract
Worldwide image geo-localization aims to infer the geographic location of an image captured anywhere on Earth, spanning street, city, regional, national, and continental scales. Existing methods rely on visual features that are sensitive to environmental variations (e.g., lighting, season, and weather) and lack effective post-processing to filter outlier candidates, limiting localization accuracy. To address these limitations, we propose DualGeo, a two-stage framework for worldwide image geo-localization. First, it establishes a geo-representational foundation by fusing image and semantic segmentation features via bidirectional cross-attention. The fused features are then aligned with GPS coordinates through dual-view contrastive learning to build a global retrieval database. Second, it performs geo-cognitive refinement by re-ranking retrieved candidates using geographic clustering. It then feeds them into large multimodal models (LMMs) for final coordinate prediction. Experiments on IM2GPS, IM2GPS3k, and YFCC4k show that DualGeo outperforms state-of-the-art methods, improving street-level (<1km) and city-level (<25km) localization accuracy by 3.6%--16.58% and 1.29%--8.77%, respectively.


## Environment

```python
# Traning on CUDA Version: 12.8
conda create -n TransGeoCLIP python=3.9
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

## Data

For the IM2GPS, IM2GPS3k, and YFCC4k datasets of the public test set, you can refer to the following links to query:
http://www.mediafire.com/

In the data directory, the metadata of the IM2GPS, IM2GPS3k, YFCC4k datasets are stored.

For the training dataset MP16-Pro dataset, you can visit the following link to query:

https://huggingface.co/datasets/Jia-py/MP16-Pro/tree/main


## Running samples

1.Training model
Run DualGeo_train.py to train the DualGeo model, if you have GPUs, you can use acc to speed it up. We recommend using fp16 for DualGeo training instead of full training. You can choose to run the command based on the number of GPUs you have, note that if your GPU memory is less than 12GB, it is recommended to reduce the training batchsize size.

```python
python DualGeo_train.py
accelerate launch --num_processes=2 --mixed_precision=fp16 DualGeo_train.py
```
2.Building index
When the training is complete, you need to build your own search database. Use the following command to build the index, in addition to providing a preliminary test code interface IndexSearch_DualGeo.py. You can build a search database for initial testing of IM2GPS, IM2GPS3k, and YFCC4k datasets.
```python
python IndexSearch_DualGeo.py
```
3.Rerank retrieval
Run the following command to reorder the index results by clustering, and you can modify the clustering radius size and the number used for clustering according to your needs.
```python
python rerank_geo_cluster.py
```
4.lmms retrieval
Run the following command to call LMMS to reorder the results and query images through API calls. You can modify the prompt to suit your needs, and of course, don't forget to replace your own api_key.
```python
python lmms.py
```



## Ref

```tex
@article{Cui2026,
  title = {{{DualGeo}}: {{A Dual-View Framework for Worldwide Image Geo-localization}},
  author = {Junchao Cui, etal.},
  date = {2026},
  title = {IEEE International Conference on Multimedia & Expo (ICME)},
  pages = {1--6}
}
```

