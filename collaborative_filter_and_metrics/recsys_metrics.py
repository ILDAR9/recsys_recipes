from typing import List, Any

from math import log2
import numpy as np


def user_hitrate(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> int:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: 1 if top-k recommendations contains at lease one relevant item
    """
    return int(len(set(y_rec[:k]).intersection(set(y_rel))) > 0)


def user_precision(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of relevant items through recommendations
    """
    return len(set(y_rel).intersection(y_rec[:k])) / k


def user_recall(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: percentage of found relevant items through recommendations
    """
    return len(set(y_rel).intersection(y_rec[:k])) / len(y_rel)


def user_ap(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: average precision metric for user recommendations
    """
    y_rel_s = set(y_rel)
    rec_count = len(y_rec)
    return 1./k * sum(user_precision(y_rel[:i], y_rec[:i], i) for i in range(1, min(rec_count, k) + 1) if y_rec[i-1] in y_rel_s)

# DCG

def compute_gain(y_value: float, gain_scheme: str) -> float:
    """
    вспомогательная функция для расчёта DCG  и NDCG, рассчитывающая показатель gain.
    Принимает на вход дополнительный аргумент - указание схемы начисления gain.

    gain_scheme: ['const', 'exp2'],
    где exp2 - (2^r−1), где r - реальная релевантность документа некоторому запросу
    """
    if gain_scheme == 'const':
        return y_value
    elif gain_scheme == 'exp2':
        return 2.**y_value - 1.
    raise NotImplementedError()


def dcg(y_rel: List[Any], y_rec: List[Any], gain_scheme: str) -> float:
    y_rel_set = set(y_rel)
    return sum(compute_gain(float(x in y_rel_set), gain_scheme) * 1./log2(i) for i, x in enumerate(y_rec, start=2))

def user_ndcg(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: ndcg metric for user recommendations
    """
    idea_dcg_val = dcg(y_rel, y_rel, gain_scheme='const')
    dcg_val = dcg(y_rel, y_rec[:k], gain_scheme = 'const')
    return dcg_val / idea_dcg_val


def user_rr(y_rel: List[Any], y_rec: List[Any], k: int = 10) -> float:
    """
    :param y_rel: relevant items
    :param y_rec: recommended items
    :param k: number of top recommended items
    :return: reciprocal rank for user recommendations
    """
    y_rel_s = set(y_rel)
    for i, x in enumerate(y_rec[:k], start=1):
        if x in y_rel_s:
            return 1./i
    return  0.


def test_precision():
    y_rel = [2, 3, 0, 8, 5, 4, 1, 9, 6, 7]
    y_rec = [1, 2, 6, 5, 3, 9, 0, 8, 4, 7]
    k = 100
    expected = 0.1
    res = user_precision(y_rel = y_rel, y_rec = y_rec, k = k)
    assert abs(res - expected) < 0.001, f'Got {res} while expected {expected}'


def test_recall():
    y_rel = [7, 2, 8, 3, 0, 9, 6, 1, 4, 5]
    y_rec = [1, 3, 6, 2, 7, 5, 8, 0, 4, 9]
    k = 1
    expected = 0.1
    res = user_recall(y_rel = y_rel, y_rec = y_rec, k = k)
    assert abs(res - expected) < 0.001, f'Got {res} while expected {expected}'

def test_ap():
    y_rel = [9,0,17,13,14,15,18,3,8,19]
    y_rec = [11,0,7,10,12,15,4,9,6,18]
    k=10
    expected = 0.1608
    res = user_ap(y_rel = y_rel, y_rec = y_rec, k = k)
    assert abs(res - expected) < 0.001, f'Got {res} while expected {expected}'

def test_ndcg():
    y_rel = [17,11,18,19,7,3,13,16,4,8]
    y_rec = [6,2,3,16,9,5,12,13,17,1]
    k = 10
    expected = 0.3405
    res = user_ndcg(y_rel = y_rel, y_rec = y_rec, k = k)
    assert abs(res - expected) < 0.001, f'Got {res} while expected {expected}'


if __name__ == '__main__':
    test_ap()
    test_ndcg()
    test_precision()
    test_recall()