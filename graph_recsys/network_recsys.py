import numpy as np
import polars as pl
from tqdm import tqdm

from typing import List, Any
from pathlib import Path

from sklearn.model_selection import train_test_split

import random
import torch

TOP_K = 20
RANDOM_STATE = 42

SUBMISSION_PATH = 'submission.parquet'


def user_intersection(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> int:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: number of items in intersection of y_rel and y_rec (truncated to top-K)
    """
    return len(set(y_rec[:k]).intersection(set(y_rel)))


def user_recall(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of found relevant items through recommendations
    """
    return user_intersection(y_rel, y_rec, k) / min(k, len(set(y_rel)))

def prepare_validation_set(data: pl.DataFrame, test_size=0.1) -> pl.DataFrame:
    # зафиксируем генератор случайных чисел
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


    # отфильтруем тех пользователей, у которых только один друг :(
    # для того, чтобы в тренировочной выборке и валидации было хотя бы по одному другу
    friends_count = data.groupby('uid').count()
    filtered_uid = set(friends_count.filter(pl.col('count') > 1)['uid'].to_list())

    data_filtered = data.filter(pl.col('uid').is_in(filtered_uid))

    # случайно выбираем ребра для валидационной выборки
    train_df, test_df = train_test_split(
        data_filtered.filter(pl.col('uid').is_in(filtered_uid)),
        stratify=data_filtered['uid'],
        test_size=test_size,
        random_state=RANDOM_STATE
    )

    return train_df, test_df

def filter_data(data: pl.DataFrame) -> pl.DataFrame:
    """
    Filter bad records
    """
    friends_count = data.groupby('uid').count()
    filtered_uid = set(friends_count.filter(pl.col('count') > 1)['uid'].to_list())

    data_filtered = data.filter(pl.col('uid').is_in(filtered_uid))
    return data_filtered


def compute_median_history_count(data: pl.DataFrame) -> int:
    grouped_df = filter_data(data).groupby('uid') \
            .agg(pl.col('friend_uid').alias('user_history'))
    

    median_seq_len = int(grouped_df['user_history'].apply(len).median())
    return median_seq_len


@torch.no_grad()
def collect_data_training(df: pl.DataFrame, model, sample_neg_count = 10, device = 'cpu'):
    
    all_uids = df['uid'].to_list()

    samples: list[np.ndarray] = []
    refs: list[int] = []

    df_grouped = df \
            .groupby('uid') \
            .agg(pl.col('friend_uid').alias('user_history'))

    model = model.to(device)
    group_len = len(df_grouped)
    for i in tqdm(range(group_len), total=group_len):
        uid, friend_uid_list = df_grouped.row(i)
        if sample_neg_count > 0:
            not_friend_uid_list = set(random.sample(all_uids, sample_neg_count)) - set(friend_uid_list + [uid])
            not_friend_uid_list = list(not_friend_uid_list)
            candidates = friend_uid_list + not_friend_uid_list
        else:
            candidates = friend_uid_list

        candidate_embs = model.forward(torch.tensor(candidates, dtype=torch.long).to(device))
        uid_emb = model.forward(torch.tensor([uid], dtype=torch.long).to(device))
        
        hadamard_values = uid_emb * candidate_embs
        samples.append(hadamard_values.cpu().numpy())
        refs += [1]*len(friend_uid_list)
        if sample_neg_count > 0:
            refs += [0]*len(not_friend_uid_list)
    
    return {
        'features': np.vstack(samples), # Hadamard product
        'y_hat': refs # yes, no
    }


def prepare_submission(data: pl.DataFrame, recs: dict[int, np.array], fpath = 'submission.parquet') -> pl.DataFrame:
    sample_submission = pl.read_parquet(Path(__file__).parent/'sample_submission.parquet')

    grouped_df = (
        sample_submission.select('uid')
        .join(
            data
            .groupby('uid')
            .agg(pl.col('friend_uid').alias('user_history')),
            'uid',
            how='left'
        )
    )

    submission = []

    for user_id, user_history in tqdm(grouped_df.rows()):
        user_history = [] if user_history is None else user_history
        
        y_rec = [uid for uid in recs[user_id] if uid not in user_history and uid != user_id]
        submission.append((user_id, y_rec[:TOP_K+10]))
        
    submission = pl.DataFrame(submission, schema=['user_id', 'y_recs'])
    submission.write_parquet(fpath)
    return submission