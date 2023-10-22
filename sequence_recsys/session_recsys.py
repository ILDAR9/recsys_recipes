import pandas as pd
import polars as pl
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
import random
from typing import List, Any
import optuna
from gensim.models import Word2Vec

SUBMITION_FILE = 'submission.parquet'
TOP_K = 20


def user_hitrate(y_relevant: List[str], y_recs: List[str], k: int = TOP_K) -> int:
    return int(len(set(y_relevant).intersection(y_recs[:k])) > 0)

def user_ndcg(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: ndcg metric for user recommendations
    """
    dcg = sum([1. / np.log2(idx + 2) for idx, item in enumerate(y_rec[:k]) if item in y_rel])
    idcg = sum([1. / np.log2(idx + 2) for idx, _ in enumerate(zip(y_rel, np.arange(k)))])
    return dcg / idcg

def sparse_user_item(user_id_seq, user_itemlist_seq):
    # соберем строчки для разреженной матрицы
    rows = []
    cols = []
    values = []
    total = len(user_id_seq)
    for user_id, train_ids in tqdm(zip(user_id_seq, user_itemlist_seq),total=total):
        count_items = len(train_ids)
        rows.extend([user_id] * count_items)
        values.extend([1] * count_items)
        cols.extend(train_ids)

    user_item_data = sp.csr_matrix((values, (rows, cols)))
    return user_item_data

def evaluate_model(model, train_ids, test_ids):
    ndcg_list = []
    hitrate_list = []
    assert len(train_ids) == len(test_ids)
    for train_ids, y_rel in tqdm(zip(train_ids, test_ids), total=len(train_ids)):
        model_preds = model.predict_output_word(
            train_ids, topn=(TOP_K + len(train_ids))
        )
        if model_preds is None:
            hitrate_list.append(0)
            continue

        y_rec = [pred[0] for pred in model_preds if pred[0] not in train_ids]
        ndcg_list.append(user_ndcg(y_rel, y_rec))
        hitrate_list.append(user_hitrate(y_rel, y_rec))
    return np.mean(ndcg_list), np.mean(hitrate_list)


SEED = 42
_grouped_df = None

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def objective(trial):
    global _grouped_df
    sg = trial.suggest_categorical('sg', [0, 1])
    window = trial.suggest_int('window', 1, 10)
    ns_exponent = trial.suggest_float('ns_exponent', -3, 3)
    negative = trial.suggest_int('negative', 3, 20)
    min_count = trial.suggest_int('min_count', 0, 20)
    vector_size = trial.suggest_categorical('vector_size', [16, 32, 64, 128])
    
    print({
        'sg': sg,
        'window_len': window,
        'ns_exponent': ns_exponent,
        'negative': negative,
        'min_count': min_count,
        'vector_size': vector_size,
    })
    
    set_seed(SEED)
    model = Word2Vec(
        _grouped_df['train_item_ids'].to_list(),
        window=window,
        sg=sg,
        hs=0,
        min_count=min_count,
        vector_size=vector_size,
        negative=negative,
        ns_exponent=ns_exponent,
        seed=SEED,
        epochs=10,
    )
    
    mean_ndcg, mean_hitrate = evaluate_model(model,
                                            _grouped_df['train_item_ids'],
                                            _grouped_df['test_item_ids'])
    print(f'NDCG@{TOP_K} = {mean_ndcg:.4f} Hitrate@{TOP_K} = {mean_hitrate:.4f}')
    return mean_ndcg
    
    
def optimise_hyperparams(grouped_df):
    global _grouped_df
    _grouped_df = grouped_df
    study = optuna.create_study(directions=('maximize',))
    study.optimize(objective, n_trials=100)

    print('Best params:', study.best_params)
    return study

def prepare_submission(data, model,
                       artist_mapping: dict[str, int],
                       artist_mapping_inverse: dict[int, str],
                       top_artist_ids: list[str]):
    set_seed(SEED)
    submission = []
    top_artist_ids: list[int] = list(map(artist_mapping.get, top_artist_ids))
    count_empty = 0
    for user_id, user_history in tqdm(data.groupby('user_id').agg(pl.col('artist_id')).rows()):
        history_ids = list(map(artist_mapping.get, user_history))
        model_preds = model.predict_output_word(
            history_ids, topn=(TOP_K + len(history_ids))
        )
        if model_preds is None:
            count_empty += 1
            print(f'model_preds is empty, {count_empty}')
            y_rec: list[int] = top_artist_ids.copy()
        else:
            y_rec: list[int] = [p[0] for p in model_preds]
        y_rec: list[str] = [artist_mapping_inverse[artist_id] for artist_id in y_rec if artist_id not in set(history_ids)]
        submission.append((user_id, y_rec[:TOP_K]))
        
    submission = pl.DataFrame(submission, schema=('user_id', 'y_rec'))
    submission.write_parquet(SUBMITION_FILE)
    return submission