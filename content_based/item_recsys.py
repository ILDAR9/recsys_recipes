import os
import numpy as np
from tqdm import tqdm

from typing import List

import numpy as np
import scipy.sparse as sp
import polars as pl
from sklearn.preprocessing import normalize

from PIL import Image
import requests
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorboard.plugins import projector
import pandas as pd

TOP_K = 10
SUBMISSION_PATH = 'submission.parquet'
RELEVANT_TITLES_PATH = 'relevant_titles_subsample.parquet'

# FRAME_TYPE = pl.dataframe.frame.DataFrame
FRAME_TYPE = pd.DataFrame

def hitrate(y_relevant: list[str], y_preds: list[str], k: int = TOP_K) -> int:
    return int(len(set(y_relevant).intersection(y_preds[:k])) > 0)

# код для подсчета метрики качества
def print_score() -> None:
    hitrate_list = []
    user_preds = {title_id: recs for title_id, recs in pl.read_parquet(SUBMISSION_PATH).rows()}
    for title_id, relevant_items in pl.read_parquet(RELEVANT_TITLES_PATH).rows():
        recommended_titles = user_preds.get(title_id, [])[:TOP_K]
        hitrate_list.append(hitrate(relevant_items, recommended_titles))

    mean_hitrate = float(np.mean(hitrate_list))
    print(f'HITRATE@10 = {mean_hitrate}')

def prepare_cluster_vis(data: np.ndarray, movies_df: FRAME_TYPE, log_dir = './embs') -> None:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        for title_name in movies_df[:1_000]['name']:
            f.write(f'{title_name}\n')

    weights = tf.Variable(data)
    checkpoint = tf.train.Checkpoint(embedding=weights)
    checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, config)

def normalise_data(sparse_data: np.ndarray) -> np.ndarray:
    return normalize(sparse_data, norm='l2', axis=1)

def similarities_from_sparse(sparse_data: sp._csr.csr_matrix) -> np.ndarray:
    similarities = (sparse_data @ sparse_data.T).A
    # уберем 1 по диагонали, чтобы не рекомендовать тайтл для самого себя
    similarities -= np.eye(len(similarities), dtype=similarities.dtype)
    return similarities

def get_recommendations(similarities: np.ndarray, movies_df: pd.DataFrame,  title_ind: int, k: int = 10):
    nearest_inds = np.argsort(similarities[title_ind])[::-1][:k]
    return movies_df.iloc[nearest_inds]['title_id'].to_list()

def get_recommendations_inds(similarities: np.ndarray, item_ind: int, k: int = 10) -> np.array:
    nearest_inds = np.argsort(similarities[item_ind])[::-1][:k]
    return nearest_inds

def vis_recs(similarities, movies_df: pd.DataFrame, title_id: str, k = 5):
    fig, axs = plt.subplots(1, k + 1, figsize=(25, 10))

    title_ind = movies_df['title_id'].to_list().index(title_id)

    relevant_titles = (
        pl.read_parquet(RELEVANT_TITLES_PATH)
        .filter(pl.col('title_id') == title_id)
    )['relevant_titles'].explode().to_list()

    # отрисовываем запрашиваемый тайтл
    url = movies_df['poster_url'][title_ind]
    im = Image.open(requests.get(url, stream=True).raw)
    axs[0].imshow(im)
    axs[0].axis('off')
    axs[0].set_title(movies_df['name'][title_ind])

    # строим рекомендации
    nearest_inds = get_recommendations_inds(similarities,title_ind, k)
    recs_posters = movies_df.iloc[nearest_inds]['poster_url'].values
    recs_names = movies_df.iloc[nearest_inds]['name'].values
    recs_title_ids = movies_df.iloc[nearest_inds]['title_id'].values

    # визуализируем рекомендации
    for i, (url, name) in enumerate(zip(recs_posters, recs_names)):
        im = Image.open(requests.get(url, stream=True).raw)
        axs[1 + i].imshow(im)
        axs[1 + i].axis('off')
        axs[1 + i].set_title(name, color=('g' if recs_title_ids[i] in relevant_titles else 'r'))

    plt.show()

def prepare_submission(movies_df: pd.DataFrame, similarities: np.ndarray) -> None:
    submission = []
    for title_ind in tqdm(range(len(movies_df))):
        title_id = movies_df['title_id'][title_ind]
        recommended_titles = get_recommendations(similarities, movies_df, title_ind, TOP_K)
        submission.append((title_id, recommended_titles))

    submission = pl.DataFrame(submission, schema=('title_id', 'recs'))
    submission.write_parquet(SUBMISSION_PATH)

    print_score()