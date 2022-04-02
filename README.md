# CS4243 Project Group 21

Predicting popularity of images with reddit community scores

## Requirements
* **Reddit Crawler** (to download data off reddit)
  * [praw](https://praw.readthedocs.io/en/stable/) 7.5.0
  * Pillow 9.0.1
  * Requests 2.27.1
  * A suitable reddit API OAuth token

## Setup
Place Reddit data under `data/reddit`

Collate Reddit dataset
```
python -m dataset.collate_reddit_data
```

Train model
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m train.train_model
```
