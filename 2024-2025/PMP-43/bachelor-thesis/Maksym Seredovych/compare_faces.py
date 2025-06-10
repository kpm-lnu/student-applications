from scipy.spatial import distance
import pywt
import numpy as np
import cv2
from feutures import compute_wavelet_features_from_coeffs_to_dict, compute_entropy
from wavlet_transfroms import get_wavelet_coefficients
from tabulate import tabulate

def compare_faces(img1, img2,stat_key, metric, wavlet):
    """Порівнює два зображення за допомогою різних метрик відстаней"""
    coeffs1 = get_wavelet_coefficients(img1,wavlet)
    coeffs2 = get_wavelet_coefficients(img2,wavlet)

    f1 = compute_wavelet_features_from_coeffs_to_dict(coeffs1,compute_entropy)
    f2 = compute_wavelet_features_from_coeffs_to_dict(coeffs2,compute_entropy)
    
    dist = compare_feature_dicts(f1,f2,stat_key,metric)

    return dist

def compare_feature_dicts(
    features1,
    features2,
    stat_key: str = 'mean',
    metric: str = 'euclidean'
) -> float:
    """
    Порівнює два списки словників (вейвлет-ознаки) за заданим ключем статистики
    і метрикою відстані.

    Параметри:
        features1, features2 : списки словників зі статистиками
        stat_key : 'mean', 'std', 'var', 'entropy'
        metric : 'euclidean', 'manhattan', 'cosine', 'chebyshev'

    Повертає:
        float : обчислена відстань або від'ємна косинусна схожість
    """
    if len(features1) != len(features2):
        raise ValueError("Feature lists must be of the same length and aligned.")

    vector1 = np.array([f[stat_key] for f in features1])
    vector2 = np.array([f[stat_key] for f in features2])

    if metric == 'euclidean':
        return np.linalg.norm(vector1 - vector2)
    elif metric == 'manhattan':
        return np.sum(np.abs(vector1 - vector2))
    elif metric == 'cosine':
        return distance.cosine(vector1, vector2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    

def compare_faces_all_stats_and_metrics(
    image1_path: str,
    image2_path: str,
    wavelet: str = 'db4',
    level: int = 3
):
    """
    Порівнює два обличчя за всіма статистиками та метриками відстані.
    Повертає словник виду:
    {
        'mean':    {'euclidean': ..., 'manhattan': ..., 'cosine': ...},
        'std':     {...},
        'var':     {...},
        'entropy': {...},
    }
    """
    coeffs1 = get_wavelet_coefficients(image1_path, wavelet, level)
    coeffs2 = get_wavelet_coefficients(image2_path, wavelet, level)

    features1 = compute_wavelet_features_from_coeffs_to_dict(coeffs1, compute_entropy)
    features2 = compute_wavelet_features_from_coeffs_to_dict(coeffs2, compute_entropy)

    stat_keys = ['mean', 'std', 'var', 'entropy']
    metrics  = ['euclidean', 'manhattan']

    results = {}

    for stat in stat_keys:
        results[stat] = {}
        for metric in metrics:
            dist = compare_feature_dicts(features1, features2, stat, metric)
            results[stat][metric] = round(dist, 4)

    return results