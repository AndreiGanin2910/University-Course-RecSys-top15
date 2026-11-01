"""Рекомендации дополнительных курсов МГПУ — pNDCG@5 = 0.4528070.

Полный рецепт лучшего отправленного решения: признаки истории и текста,
LightGBM, CatBoost, обучаемая E5-совместимость, Ridge и холодная эвристика.
Запуск: python main.py. Исходные файлы положите в data/.
"""

# ============================================================================
# 1. Импорты и настройки
# ============================================================================

from __future__ import annotations

import csv
import gc
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

# Задаём ограничения до импорта численных библиотек.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")


OUTPUT_NAME = "submission.csv"


E5_MODEL = "intfloat/multilingual-e5-large"


E5_REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"


E5_NUM_THREADS = 1  # Надёжный CPU-режим при совместном использовании с CatBoost.


PRIORITY_WEIGHT = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35}


GROUPS = (("institute", 50), ("specialty", 25))


P3_TRAIN_WEIGHT = 1.5


PRIORITY_SCALE = 0.65


NEW_ENSEMBLE_SHARE = 0.50


PRIORITY_SHARE = 0.75


CATBOOST_SHARE = 0.25


WARM_FAVORITE_BONUS = 0.05


COLD_POPULARITY_SHARE = 0.20


NEW_COURSE_PRIOR = {"strength": 0.50, "max_odds_factor": 4.0, "pseudocount": 10.0}


BASE_LGB_PARAMS = {
    "objective": "binary", "n_estimators": 600, "learning_rate": 0.035,
    "num_leaves": 31, "min_child_samples": 200, "reg_lambda": 5,
    "n_jobs": 4, "verbose": -1,
}


LARGE_LGB_PARAMS = {
    "objective": "binary", "n_estimators": 1800, "learning_rate": 0.035,
    "num_leaves": 63, "min_child_samples": 50, "reg_lambda": 5,
    "n_jobs": 2, "verbose": -1, "random_state": 42,
}


CATBOOST_PARAMS = {
    "iterations": 1000, "depth": 7, "learning_rate": 0.06,
    "l2_leaf_reg": 10, "loss_function": "Logloss", "max_ctr_complexity": 2,
    "bootstrap_type": "Bernoulli", "subsample": 0.8,
    "thread_count": 4, "random_seed": 42,
    "allow_writing_files": False, "use_best_model": False,
}


CATEGORICAL_COLUMNS = ("module_id", "institute", "specialty")


FAVORITE_PATTERN = re.compile(
    r"добавили\s+данный\s+модуль\s+в\s+избранное\s*(\d+)", re.I)


SUFFIX = re.compile(
    r"Количество обучающихся, которые добавили данный модуль в избранное\s*"
    r"\d+\s*человек[а-я]*\.?", re.I)


FAVORITE_SUFFIX = SUFFIX

# ============================================================================
# 2. Чтение данных и вспомогательные функции
# ============================================================================

def read_jsonl(filename: str) -> list[dict]:
    with open(os.path.join(DATA_DIR, filename), encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def read_answers() -> dict[str, dict[str, int]]:
    """id -> {module_id: priority} для известного периода 3."""
    with open(os.path.join(DATA_DIR, "val_solution.csv"), encoding="utf-8", newline="") as file:
        return {
            row["id"]: {
                module_id: int(priority)
                for module_id, priority in (token.split(":") for token in row["ground_truth"].split())
            }
            for row in csv.DictReader(file)
        }


def period2_examples(train: list[dict], modules: list[dict]) -> tuple[list[dict], list[dict]]:
    """Делаем задачу P1 -> P2. У студента видна только история до P2."""
    available_p2 = [m["module_id"] for m in modules if 2 in m["available_in_periods"]]
    records, answers = [], []
    for student in train:
        period2 = next((h for h in student["history"] if h["period"] == 2), None)
        if period2 is None:
            continue
        records.append({
            "id": student["student_id"],
            "profile": student["profile"],
            "history": [h for h in student["history"] if h["period"] < 2],
            "available_modules": available_p2,
        })
        answers.append({choice["module_id"]: choice["priority"] for choice in period2["choices"]})
    return records, answers


def complete_past(
    train: list[dict], val: list[dict], test: list[dict], answers_p3: dict
) -> list[dict]:
    """Одна запись на студента, все известные выборы P1-P3 без повторов."""
    merged: dict[str, dict] = {}
    for collection in (train, val, test):
        for student in collection:
            sid = student.get("student_id", student.get("id"))
            item = merged.setdefault(sid, {
                "profile": {"institute": "", "specialty": ""}, "history": {}
            })
            if student["profile"]["institute"]:
                item["profile"] = student["profile"]
            for period in student["history"]:
                number = period["period"]
                if number in item["history"]:
                    assert item["history"][number] == period["choices"]
                item["history"][number] = period["choices"]
    for sid, selected in answers_p3.items():
        choices = [
            {"module_id": module_id, "priority": priority}
            for module_id, priority in sorted(selected.items(), key=lambda item: item[1])
        ]
        item = merged[sid]
        if 3 in item["history"]:
            assert item["history"][3] == choices
        item["history"][3] = choices
    return [
        {"profile": item["profile"], "history": [
            {"period": number, "choices": choices}
            for number, choices in sorted(item["history"].items())
        ]}
        for item in merged.values()
    ]


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Внутри каждого студента: от 0 (низший ранг) почти до 1 (высший)."""
    return np.argsort(np.argsort(values, axis=1), axis=1).astype(np.float32) / values.shape[1]

# ============================================================================
# 3. Базовые, поведенческие и текстовые признаки
# ============================================================================

def features_for_period(
    records: list[dict], target_period: int, past_records: list[dict], modules: list[dict]
) -> tuple[list[str], np.ndarray, list[str]]:
    """Создаёт числовые признаки для всех пар студент x доступный курс.

    past_records содержит известные выборы только из периодов до target_period.
    Строки X идут в порядке records, внутри студента — в порядке candidates.
    """
    candidates = records[0]["available_modules"]
    assert all(r["available_modules"] == candidates for r in records)
    candidate_index = {module_id: j for j, module_id in enumerate(candidates)}
    n_students, n_candidates = len(records), len(candidates)
    features: dict[str, np.ndarray] = {}

    # 1. Частота выбора курсов и предпочтения института/направления.
    all_count, recent_count = Counter(), Counter()
    group_count = {field: defaultdict(Counter) for field in ("institute", "specialty")}
    recent_group_size = {field: Counter() for field in ("institute", "specialty")}
    recent_students = 0
    for student in past_records:
        has_recent = any(h["period"] == target_period - 1 for h in student["history"])
        if has_recent:
            recent_students += 1
            for field in recent_group_size:
                recent_group_size[field][student["profile"][field]] += 1
        for period in student["history"]:
            if period["period"] >= target_period:
                continue
            for choice in period["choices"]:
                module_id = choice["module_id"]
                all_count[module_id] += 1
                if period["period"] == target_period - 1:
                    recent_count[module_id] += 1
                    for field in group_count:
                        group_count[field][student["profile"][field]][module_id] += 1

    for name, counts in (("pop_all", all_count), ("pop_recent", recent_count)):
        values = np.array([counts[m] for m in candidates], dtype=np.float32)
        features[name] = np.broadcast_to(np.log1p(values), (n_students, n_candidates)).copy()

    global_rate = np.array(
        [recent_count[m] / max(recent_students, 1) for m in candidates], dtype=np.float32
    )
    for field, smoothing in (("institute", 50), ("specialty", 25)):
        lift = np.zeros((n_students, n_candidates), dtype=np.float32)
        for i, student in enumerate(records):
            category = student["profile"][field]
            if not category:
                continue  # пустой профиль = нейтральный признак
            counts = group_count[field][category]
            size = recent_group_size[field][category]
            category_rate = (
                np.array([counts[m] for m in candidates]) + smoothing * global_rate
            ) / (size + smoothing)
            lift[i] = np.log((category_rate + .004) / (global_rate + .004))
        features[f"{field}_lift"] = lift

    # 2. Выбирал ли сам студент этот курс и с каким приоритетом.
    repeat = np.zeros((n_students, n_candidates), dtype=np.float32)
    repeat_recent = repeat.copy()
    repeat_priority = repeat.copy()
    for i, student in enumerate(records):
        for period in student["history"]:
            if period["period"] >= target_period:
                continue
            for choice in period["choices"]:
                j = candidate_index.get(choice["module_id"])
                if j is None:
                    continue
                repeat[i, j] = 1
                repeat_priority[i, j] = max(
                    repeat_priority[i, j], PRIORITY_WEIGHT[choice["priority"]]
                )
                if period["period"] == target_period - 1:
                    repeat_recent[i, j] = 1
    features.update(repeat=repeat, repeat_recent=repeat_recent, repeat_priority=repeat_priority)

    # 3. Сходство полного описания курса с прежними выборами студента.
    all_ids = [m["module_id"] for m in modules]
    all_index = {module_id: j for j, module_id in enumerate(all_ids)}
    clean_descriptions = [FAVORITE_SUFFIX.sub("", m["description"]) for m in modules]
    full_texts = [m["title"] + " " + description
                  for m, description in zip(modules, clean_descriptions)]
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True
    )
    word_vectors = word_vectorizer.fit_transform(full_texts)
    candidate_words = word_vectors[[all_index[m] for m in candidates]]
    sim_mean = np.zeros((n_students, n_candidates), dtype=np.float32)
    sim_max = np.zeros_like(sim_mean)
    for i, student in enumerate(records):
        previous = [choice["module_id"] for h in student["history"]
                    if h["period"] < target_period for choice in h["choices"]]
        if previous:
            similarity = (
                candidate_words @ word_vectors[[all_index[m] for m in previous]].T
            ).toarray()
            sim_mean[i] = similarity.mean(axis=1)
            sim_max[i] = similarity.max(axis=1)
    features.update(sim_mean=sim_mean, sim_max=sim_max)

    # 4. Сходство коротких названий и названия курса с профилем.
    profile_texts = sorted({
        r["profile"]["specialty"] + " " + r["profile"]["institute"]
        for r in records if r["profile"]["specialty"]
    })
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=50000, sublinear_tf=True,
    )
    char_vectors = char_vectorizer.fit_transform(
        [m["title"] for m in modules] + profile_texts
    )
    candidate_titles = char_vectors[[all_index[m] for m in candidates]]
    profile_index = {text: len(modules) + j for j, text in enumerate(profile_texts)}
    title_mean = np.zeros((n_students, n_candidates), dtype=np.float32)
    title_max = np.zeros_like(title_mean)
    profile_title = np.zeros_like(title_mean)
    for i, student in enumerate(records):
        previous = [choice["module_id"] for h in student["history"]
                    if h["period"] < target_period for choice in h["choices"]]
        if previous:
            similarity = (
                candidate_titles @ char_vectors[[all_index[m] for m in previous]].T
            ).toarray()
            title_mean[i] = similarity.mean(axis=1)
            title_max[i] = similarity.max(axis=1)
        profile_text = student["profile"]["specialty"] + " " + student["profile"]["institute"]
        if profile_text in profile_index:
            profile_title[i] = (
                candidate_titles @ char_vectors[profile_index[profile_text]].T
            ).toarray().ravel()
    features.update(title_mean=title_mean, title_max=title_max, profile_title=profile_title)

    # 5. Совместная встречаемость курсов в пятёрках прошлого периода.
    cooccurrence = np.zeros((len(modules), len(modules)), dtype=np.float32)
    frequency = np.zeros(len(modules), dtype=np.float32)
    for student in past_records:
        for period in student["history"]:
            if period["period"] >= target_period:
                continue
            chosen = [all_index[c["module_id"]] for c in period["choices"]]
            frequency[chosen] += 1
            for a in chosen:
                for b in chosen:
                    if a != b:
                        cooccurrence[a, b] += 1
    normalized_cooccurrence = cooccurrence / np.sqrt(
        np.outer(frequency + 1, frequency + 1)
    )
    co_mean = np.zeros((n_students, n_candidates), dtype=np.float32)
    co_max = np.zeros_like(co_mean)
    candidate_ids = [all_index[m] for m in candidates]
    for i, student in enumerate(records):
        previous = [all_index[c["module_id"]] for h in student["history"]
                    if h["period"] < target_period for c in h["choices"]]
        if previous:
            similarity = normalized_cooccurrence[np.ix_(candidate_ids, previous)]
            co_max[i] = similarity.max(axis=1)
            co_mean[i] = similarity.mean(axis=1)
    features.update(co_max=co_max, co_mean=co_mean)

    # 6. Простые свойства курса и длина истории студента.
    module_by_id = {m["module_id"]: m for m in modules}
    module_features = {
        "available_prior_periods": [
            sum(p < target_period for p in module_by_id[m]["available_in_periods"])
            for m in candidates
        ],
        "title_length": [len(module_by_id[m]["title"]) for m in candidates],
        "description_length": [
            len(FAVORITE_SUFFIX.sub("", module_by_id[m]["description"]))
            for m in candidates
        ],
    }
    for name, values in module_features.items():
        features[name] = np.broadcast_to(
            np.array(values, dtype=np.float32), (n_students, n_candidates)
        ).copy()
    history_size = np.array([len(r["history"]) for r in records], dtype=np.float32)
    features["history_periods"] = np.broadcast_to(
        history_size[:, None], (n_students, n_candidates)
    ).copy()

    feature_names = list(features)
    matrix = np.stack([features[name] for name in feature_names], axis=-1)
    return candidates, matrix.reshape(-1, len(feature_names)), feature_names


def behavior_features(records, period, source, candidates, modules):
    """Девять признаков прежней доступности, пропусков и давности популярности."""
    n, k = len(records), len(candidates)
    module_by_id = {m["module_id"]: m for m in modules}
    available = {mid: set(module_by_id[mid]["available_in_periods"]) for mid in candidates}
    chosen = np.zeros((n, k), np.float32)
    opportunity = np.zeros((n, k), np.float32)
    recent_priority = np.zeros((n, k), np.float32)
    recent_skipped = np.zeros((n, k), np.float32)
    older_priority = np.zeros((n, k), np.float32)
    for i, record in enumerate(records):
        for past in record["history"]:
            if past["period"] >= period:
                continue
            choice = {c["module_id"]: c["priority"] for c in past["choices"]}
            for j, mid in enumerate(candidates):
                if past["period"] in available[mid]:
                    opportunity[i, j] += 1
                    if past["period"] == period - 1 and mid not in choice:
                        recent_skipped[i, j] = 1
                if mid in choice:
                    chosen[i, j] += 1
                    if past["period"] == period - 1:
                        recent_priority[i, j] = PRIORITY_WEIGHT[choice[mid]]
                    else:
                        older_priority[i, j] = max(
                            older_priority[i, j], PRIORITY_WEIGHT[choice[mid]]
                        )

    # Популярность в последний период доступности, в том числе после перерыва.
    # Деление на число студентов учитывает разный размер прошлых периодов.
    choice_by_period = defaultdict(Counter)
    student_n_by_period = Counter()
    for record in source:
        for past in record["history"]:
            if past["period"] >= period:
                continue
            student_n_by_period[past["period"]] += 1
            choice_by_period[past["period"]].update(
                c["module_id"] for c in past["choices"]
            )
    last_rate = np.zeros(k, np.float32)
    last_lag = np.zeros(k, np.float32)
    previous_rate = np.zeros(k, np.float32)
    for j, mid in enumerate(candidates):
        previous_periods = sorted(p for p in available[mid] if p < period)
        if previous_periods:
            recent = previous_periods[-1]
            last_rate[j] = choice_by_period[recent][mid] / max(student_n_by_period[recent], 1)
            last_lag[j] = period - recent
        if len(previous_periods) >= 2:
            earlier = previous_periods[-2]
            previous_rate[j] = choice_by_period[earlier][mid] / max(student_n_by_period[earlier], 1)

    item_rate = np.broadcast_to(np.log1p(1000 * last_rate), (n, k)).copy()
    item_previous = np.broadcast_to(np.log1p(1000 * previous_rate), (n, k)).copy()
    item_lag = np.broadcast_to(last_lag, (n, k)).copy()
    return {
        "student_prior_opportunities": opportunity,
        "student_prior_chooses": chosen,
        "student_prior_skip_count": opportunity - chosen,
        "student_recent_skip": recent_skipped,
        "student_recent_priority": recent_priority,
        "student_older_priority": older_priority,
        "last_offered_popularity": item_rate,
        "earlier_offered_popularity": item_previous,
        "last_offered_lag": item_lag,
    }


def text_similarity(modules):
    """Две матрицы сходства курсов: по названиям и по очищенному полному тексту."""
    titles = [m["title"] for m in modules]
    descriptions = [
        re.sub(
            r"Количество обучающихся, которые добавили данный модуль в избранное\s*\d+\s*человек[а-я]*\.?",
            "", m["description"], flags=re.I,
        )
        for m in modules
    ]
    title_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=50000, sublinear_tf=True,
    ).fit_transform(titles)
    content_vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=30000,
        sublinear_tf=True,
    ).fit_transform([a + " " + b for a, b in zip(titles, descriptions)])
    return (title_vec @ title_vec.T).toarray().astype(np.float32), \
           (content_vec @ content_vec.T).toarray().astype(np.float32)


def peer_content_features(records, period, source, candidates, modules,
                          similarities=None, exclude_self=False):
    """Перенос прошлой популярности с похожих курсов в разрезах профиля."""
    title_sim, content_sim = similarities if similarities is not None else text_similarity(modules)
    all_ids = [m["module_id"] for m in modules]
    index = {mid: j for j, mid in enumerate(all_ids)}
    candidate_index = np.array([index[mid] for mid in candidates])
    n, k = len(records), len(candidates)

    recent_count = np.zeros(len(modules), np.float32)
    group_count = {field: defaultdict(Counter) for field, _ in GROUPS}
    group_size = {field: Counter() for field, _ in GROUPS}
    recent_students = 0
    for student in source:
        history = next((h for h in student["history"] if h["period"] == period - 1), None)
        if history is None:
            continue
        recent_students += 1
        chosen = [c["module_id"] for c in history["choices"]]
        for mid in chosen:
            recent_count[index[mid]] += 1
        for field, _ in GROUPS:
            key = student["profile"][field]
            if key:
                group_size[field][key] += 1
                group_count[field][key].update(chosen)

    global_rate = recent_count / max(recent_students, 1)
    observed = recent_count > 0
    kernels = {}
    for name, similarity in (("title", title_sim), ("content", content_sim)):
        kernel = similarity[candidate_index].copy()
        if exclude_self:
            kernel[np.arange(k), candidate_index] = 0
        kernel[:, ~observed] = 0
        kernel **= 4
        kernel /= np.maximum(kernel.sum(axis=1, keepdims=True), 1e-8)
        kernels[name] = kernel

    result = {}
    for name, kernel in kernels.items():
        global_transferred = kernel @ global_rate
        for field, shrink in GROUPS:
            keys = sorted({r["profile"][field] for r in records if r["profile"][field]})
            group_predictions = np.zeros((len(keys), k), np.float32)
            for pos, key in enumerate(keys):
                choices = group_count[field][key]
                counts = np.array([choices[mid] for mid in all_ids], np.float32)
                smoothed = (counts + shrink * global_rate) / (group_size[field][key] + shrink)
                group_predictions[pos] = kernel @ smoothed
            lookup = {key: j for j, key in enumerate(keys)}
            transferred = np.broadcast_to(global_transferred, (n, k)).copy()
            for i, record in enumerate(records):
                j = lookup.get(record["profile"][field])
                if j is not None:
                    transferred[i] = group_predictions[j]
            result[f"peer_{field}_{name}_demand"] = np.log1p(1000 * transferred)
            result[f"peer_{field}_{name}_lift"] = np.log(
                (transferred + .004) / (global_transferred[None] + .004)
            )
    return result


def with_behavior(records, period, source, modules):
    """Объединяет 18 базовых и девять поведенческих признаков."""
    candidates, base, names = features_for_period(records, period, source, modules)
    extra = behavior_features(records, period, source, candidates, modules)
    extended = np.hstack([base] + [extra[name].reshape(-1, 1) for name in extra])
    return candidates, base, extended, names, names + list(extra)


def peer_features(records, period, source, modules, similarities):
    """Добавляет восемь групповых текстовых признаков: всего 35 столбцов."""
    candidates, _, behavior, _, names = with_behavior(records, period, source, modules)
    extra = peer_content_features(records, period, source, candidates, modules, similarities)
    full = np.column_stack([behavior] + [value.ravel() for value in extra.values()])
    return candidates, full, names + list(extra)

# ============================================================================
# 4. Полные E5-векторы и семантические признаки
# ============================================================================

def get_embeddings(modules):
    """Считает E5-векторы названий и описаний курсов.

    Фиксируем ревизию, CPU и размер батча исходного решения. Фраза с
    количеством добавлений в избранное удаляется перед кодированием.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(E5_NUM_THREADS)
    try:
        model = SentenceTransformer(
            E5_MODEL, revision=E5_REVISION, local_files_only=True, device="cpu")
    except OSError:
        # Первый запуск на новом компьютере скачивает только веса E5.
        model = SentenceTransformer(E5_MODEL, revision=E5_REVISION, device="cpu")
    texts = ["query: " + item["title"] + ". "
             + SUFFIX.sub("", item["description"])
             for item in modules]
    vectors = model.encode(
        texts, batch_size=8, normalize_embeddings=True,
        show_progress_bar=True).astype(np.float32)
    del model
    gc.collect()
    return vectors


def semantic_features(records, period, source, candidates, modules, vectors):
    """Восемь семантических признаков; при переносе спроса исключаем сам курс."""
    ids = [m["module_id"] for m in modules]
    idx = {mid: i for i, mid in enumerate(ids)}
    ci = np.array([idx[mid] for mid in candidates])
    sim = (vectors @ vectors.T).astype(np.float32)
    np.clip(sim, -1, 1, out=sim)
    n, k = len(records), len(candidates)
    recent = Counter()
    group = {name: defaultdict(Counter) for name in ("institute", "specialty")}
    sizes = {name: Counter() for name in group}
    recent_students = 0
    for student in source:
        previous = next((h for h in student["history"] if h["period"] == period - 1), None)
        if previous is None:
            continue
        recent_students += 1
        choice_ids = [c["module_id"] for c in previous["choices"]]
        recent.update(choice_ids)
        for name in group:
            key = student["profile"][name]
            if key:
                sizes[name][key] += 1
                group[name][key].update(choice_ids)
    counts = np.array([recent[mid] for mid in ids], np.float32)
    rates = counts / max(recent_students, 1)
    kernel = sim[ci].copy()
    # Исключаем прошлые выборы самого кандидата из семантического переноса.
    # Тогда признак работает и для курсов, которых прежде не было.
    kernel[np.arange(k), ci] = -2
    kernel[:, counts == 0] = -2
    kernel = np.exp(np.clip((kernel - kernel.max(axis=1, keepdims=True)) / .06, -40, 0))
    kernel[:, counts == 0] = 0
    kernel[np.arange(k), ci] = 0
    kernel /= np.maximum(kernel.sum(axis=1, keepdims=True), 1e-8)
    global_demand = kernel @ rates
    result = {"semantic_global_demand": np.broadcast_to(
        np.log1p(1000 * global_demand), (n, k)).copy()}
    for name, shrink in (("institute", 50), ("specialty", 25)):
        keys = sorted({r["profile"][name] for r in records if r["profile"][name]})
        table = np.zeros((len(keys), k), np.float32)
        for row, key in enumerate(keys):
            own_counts = np.array([group[name][key][mid] for mid in ids], np.float32)
            group_rates = (own_counts + shrink * rates) / (sizes[name][key] + shrink)
            table[row] = kernel @ group_rates
        lookup = {key: row for row, key in enumerate(keys)}
        demand = np.broadcast_to(global_demand, (n, k)).copy()
        for i, record in enumerate(records):
            j = lookup.get(record["profile"][name])
            if j is not None:
                demand[i] = table[j]
        result[f"semantic_{name}_demand"] = np.log1p(1000 * demand)
        result[f"semantic_{name}_lift"] = np.log((demand + .004) / (global_demand[None] + .004))
    mean = np.zeros((n, k), np.float32)
    peak = np.zeros_like(mean)
    weighted = np.zeros_like(mean)
    for i, record in enumerate(records):
        past = [(idx[c["module_id"]], (6 - c["priority"]) / (period - h["period"]))
                for h in record["history"] if h["period"] < period for c in h["choices"]]
        if not past:
            continue
        old, weights = map(np.array, zip(*past))
        old = old.astype(int)
        values = sim[np.ix_(ci, old)]
        distinct = ci[:, None] != old[None, :]
        values = values * distinct
        denom = np.maximum(distinct.sum(axis=1), 1)
        mean[i] = values.sum(axis=1) / denom
        peak[i] = values.max(axis=1)
        weighted[i] = (values * weights).sum(axis=1) / np.maximum(
            (distinct * weights).sum(axis=1), 1e-8)
    result["semantic_history_mean"] = mean
    result["semantic_history_max"] = peak
    result["semantic_history_priority_mean"] = weighted
    return result

# ============================================================================
# 5. Темы курсов, языки и интересы студента
# ============================================================================

TOPICS = {
    "psychology": r"психолог|эмоц|чувств|отношен|самопозн|саморазвит|psycholog|emotional|emotion|feeling",
    "education": r"педагог|учител|препода|обучен|урок|школ|teach|educat|school|lesson",
    "law": r"правов|юрид|закон|уголов|судеб|law\b|legal",
    "business": r"эконом|финанс|бизнес|менедж|маркетинг|предприним|управлен|business|financ|marketing|management",
    "technology": r"цифров|программир|компьютер|данные|данных|данными|робот|нейросет|искусственн.{0,15}интеллект|digital|computer|data|programming|artificial intelligence",
    "art": r"искусств|дизайн|рисован|живопис|худож|декоратив|рукодели|арт\b|art\b|design|craft",
    "film_media": r"кино|фильм|медиа|видео|сценар|режисс|cinema|film|media|video",
    "literature": r"литератур|писател|поэз|чтени|книг|literat|fiction|writers|reading",
    "music": r"музык|песн|вокал|music|song|lyrics|singing",
    "health_sport": r"здоров|спорт|физическ|медицин|анатом|болезн|health|sport|medical",
    "history": r"истор|археолог|эпох|век\b|history|historic|century",
    "culture": r"культур|традиц|межкультур|culture|cultural|tradition",
    "travel": r"туризм|турист|географ|страновед|travel|touris|geograph",
    "linguistics": r"лингвист|язык|перевод|грамматик|language|translation|linguist|grammar",
    "philosophy": r"философ|религи|этик|духовн|philosoph|religio|ethic",
    "communication": r"оратор|публичн.{0,20}выступ|коммуникац|риторик|общен|communication|speech|presentation",
}


LANGUAGES = {
    "english": r"английск|англоязыч|\benglish\b",
    "french": r"французск|\bfrench\b",
    "german": r"немецк|\bgerman\b",
    "japanese": r"японск|\bjapan(?:ese)?\b",
    "chinese": r"китайск|\bchines(?:e)?\b",
    "russian": r"русск.{0,15}язык|русск.{0,15}литератур|\brussian\b",
    "spanish": r"испанск|\bspanish\b",
    "arabic": r"арабск|\barabic\b",
}


RULES = {**TOPICS, **LANGUAGES}


PATTERNS = {name: re.compile(pattern, re.I) for name, pattern in RULES.items()}


DESCRIPTION_OVERRIDES = {
    # Слово «язык» в описании бывает метафорой, поэтому правило уточнено.
    "linguistics": re.compile(
        r"лингвист|перевод|грамматик|изучени.{0,25}язык|иностранн.{0,20}язык|"
        r"языков|английск|французск|японск.{0,15}язык|китайск.{0,15}язык|"
        r"русск.{0,15}язык|language|translation|linguist|grammar", re.I),
}


NAMES = tuple(RULES)


TOPIC_COUNT = len(TOPICS)


def tag_courses(modules):
    """Тема в названии получает 2, только в описании — 1, иначе 0."""
    tagged = np.zeros((len(modules), len(NAMES)), np.float32)
    for i, item in enumerate(modules):
        title = item["title"]
        description = SUFFIX.sub("", item["description"])[:900]
        for j, name in enumerate(NAMES):
            if PATTERNS[name].search(title):
                tagged[i, j] = 2
            elif DESCRIPTION_OVERRIDES.get(name, PATTERNS[name]).search(description):
                tagged[i, j] = 1
    return tagged


def interest_features(records, period, candidates, modules):
    """53 признака: темы/языки курса, совпадения с историей и пять сводных."""
    ids = [item["module_id"] for item in modules]
    index = {mid: i for i, mid in enumerate(ids)}
    ci = np.array([index[mid] for mid in candidates])
    course_tags = tag_courses(modules)
    candidate_tags = course_tags[ci]
    n, k, t = len(records), len(candidates), len(NAMES)
    history_affinity = np.zeros((n, t), np.float32)
    recent_affinity = np.zeros_like(history_affinity)
    profile_tags = np.zeros_like(history_affinity)
    for i, record in enumerate(records):
        total = recent_total = 0.
        for history in record["history"]:
            if history["period"] >= period:
                continue
            for choice in history["choices"]:
                weight = (6 - choice["priority"]) / (period - history["period"])
                tags = course_tags[index[choice["module_id"]]] / 2
                history_affinity[i] += weight * tags
                total += weight
                if history["period"] == period - 1:
                    recent_affinity[i] += weight * tags
                    recent_total += weight
        history_affinity[i] /= max(total, 1)
        recent_affinity[i] /= max(recent_total, 1)
        profile = record["profile"]["institute"] + " " + record["profile"]["specialty"]
        if profile.strip():
            profile_tags[i] = [bool(PATTERNS[name].search(profile)) for name in NAMES]

    # Индикаторы курса описывают его темы. Произведения с профилем интересов
    # описывают предпочтения конкретного студента.
    flag = np.broadcast_to(candidate_tags[None], (n, k, t))
    matched = flag * history_affinity[:, None]
    topic = candidate_tags[:, :TOPIC_COUNT] / 2
    language = candidate_tags[:, TOPIC_COUNT:] / 2
    topic_affinity = history_affinity[:, :TOPIC_COUNT]
    language_affinity = history_affinity[:, TOPIC_COUNT:]
    profile_match = (flag * profile_tags[:, None]).sum(axis=2) / np.maximum(
        (flag > 0).sum(axis=2), 1)
    topic_match = (topic[None] * topic_affinity[:, None]).sum(axis=2) / np.maximum(
        topic.sum(axis=1)[None], 1)
    recent_match = (candidate_tags[None] * recent_affinity[:, None]).sum(axis=2) / np.maximum(
        candidate_tags.sum(axis=1)[None], 1)
    language_match = (language[None] * language_affinity[:, None]).sum(axis=2) / np.maximum(
        language.sum(axis=1)[None], 1)
    language_conflict = (language.sum(axis=1)[None] > 0) * (
        language_affinity.sum(axis=1)[:, None] > 0) * (language_match == 0)
    summary = np.stack([profile_match, topic_match, recent_match,
                        language_match, language_conflict], axis=2)
    matrix = np.concatenate([flag, matched, summary], axis=2).astype(np.float32)
    names = ([f"course_{name}" for name in NAMES]
             + [f"interest_{name}" for name in NAMES]
             + ["profile_match", "topic_match", "recent_match",
                "language_match", "language_conflict"])
    assert matrix.shape == (n, k, len(names))
    return matrix, names


def priority_labels(records, candidates, answers):
    """Метки содержат вес истинного приоритета; невыбранный курс имеет вес ноль."""
    out = np.zeros((len(records), len(candidates)), np.float32)
    index = {mid: j for j, mid in enumerate(candidates)}
    for i, record in enumerate(records):
        for mid, priority in answers[record["id"]].items():
            out[i, index[mid]] = PRIORITY_WEIGHT[priority]
    return out.ravel()


def feature_matrix(records, period, source, modules, similarities, vectors):
    """35 peer-признаков плюс восемь E5-признаков: всего 43 столбца."""
    candidates, peer, names = peer_features(records, period, source, modules, similarities)
    extra = semantic_features(records, period, source, candidates, modules, vectors)
    extended = np.column_stack([peer] + [x.ravel() for x in extra.values()])
    assert extended.shape[1] == 43
    return candidates, peer, extended, names + list(extra)

# ============================================================================
# 6. Соседи по истории выбора и итоговые 116 признаков
# ============================================================================

NEIGHBOR_COUNTS = (50, 200)


HISTORY_NEIGHBOR_NAMES = tuple(
    f"history_neighbors_{count}_{kind}"
    for count in NEIGHBOR_COUNTS
    for kind in ("recent_rate", "recent_lift", "priority_rate", "first_rate",
                 "effective_count", "mean_similarity")
)


HISTORY_SEMANTIC_NAMES = tuple(
    f"history_neighbors_{count}_semantic_{kind}"
    for count in NEIGHBOR_COUNTS
    for kind in ("rate", "lift", "priority_rate", "first_rate")
)


def student_id(record):
    return record.get("student_id", record.get("id"))


def merge_past_with_ids(train, val, test, answers3):
    """Объединяет P1–P3 с ID, чтобы исключить самого студента из соседей."""
    merged = {}
    for records in (train, val, test):
        for record in records:
            sid = student_id(record)
            item = merged.setdefault(sid, {
                "id": sid, "profile": record["profile"], "periods": {}})
            if record["profile"]["institute"]:
                item["profile"] = record["profile"]
            for history in record["history"]:
                period = history["period"]
                assert period < 4
                choices = {c["module_id"]: c["priority"] for c in history["choices"]}
                if period in item["periods"]:
                    assert item["periods"][period] == choices
                item["periods"][period] = choices
    for sid, choices in answers3.items():
        if 3 in merged[sid]["periods"]:
            assert merged[sid]["periods"][3] == choices
        merged[sid]["periods"][3] = choices
    result = []
    for item in merged.values():
        histories = [{"period": period, "choices": [
            {"module_id": mid, "priority": priority}
            for mid, priority in sorted(choices.items(), key=lambda x: x[1])
        ]} for period, choices in sorted(item["periods"].items())]
        result.append({"id": item["id"], "profile": item["profile"],
                       "history": histories})
    assert len(result) == len({r["id"] for r in result})
    return result


def history_matrix(records, period, index):
    rows, columns, values = [], [], []
    for i, record in enumerate(records):
        profile = {}
        for history in record["history"]:
            if history["period"] >= period:
                continue
            for choice in history["choices"]:
                j = index[choice["module_id"]]
                value = PRIORITY_WEIGHT[choice["priority"]] / (period - history["period"])
                profile[j] = max(profile.get(j, 0.), value)
        for j, value in profile.items():
            rows.append(i)
            columns.append(j)
            values.append(value)
    return csr_matrix((np.array(values, np.float32), (rows, columns)),
                      shape=(len(records), len(index)))


def normalize_sparse(matrix):
    length = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return matrix.multiply((1 / np.maximum(length, 1e-8))[:, None]).tocsr()


def history_neighbor_features(records, period, source, candidates, modules, semantic_vectors=None):
    """20 признаков: голоса 50/200 соседей и их перенос на похожие курсы.

    Используем только историю до целевого периода; собственный голос
    студента исключён и из сходства, и из сглаживающего среднего.
    """
    previous = []
    neighbors = []
    for record in source:
        history = next((h for h in record["history"]
                        if h["period"] == period - 1), None)
        if history is not None:
            neighbors.append(record)
            previous.append(history["choices"])
    assert len(neighbors) == len({student_id(r) for r in neighbors})
    index = {item["module_id"]: i for i, item in enumerate(modules)}
    source_history = history_matrix(neighbors, period, index)
    query_history = history_matrix(records, period, index)
    document_count = np.asarray((source_history > 0).sum(axis=0)).ravel()
    # Редкий общий курс сильнее отличает похожих студентов: используем IDF.
    idf = np.log((len(neighbors) + 1) / (document_count + 1)) + 1
    source_history = normalize_sparse(source_history.multiply(idf))
    query_history = normalize_sparse(query_history.multiply(idf))
    similarity = (query_history @ source_history.T).toarray().astype(np.float32)
    np.clip(similarity, 0, 1, out=similarity)
    lookup = {student_id(record): i for i, record in enumerate(neighbors)}
    own = np.array([lookup.get(student_id(record), -1) for record in records])
    own_exists = own >= 0
    similarity[np.flatnonzero(own_exists), own[own_exists]] = 0
    assert np.all(similarity[np.flatnonzero(own_exists), own[own_exists]] == 0)

    candidate_index = {mid: j for j, mid in enumerate(candidates)}
    binary = np.zeros((len(neighbors), len(candidates)), np.float32)
    priority = np.zeros_like(binary)
    first = np.zeros_like(binary)
    for i, choices in enumerate(previous):
        for choice in choices:
            j = candidate_index.get(choice["module_id"])
            if j is not None:
                binary[i, j] = 1
                priority[i, j] = PRIORITY_WEIGHT[choice["priority"]]
                first[i, j] = choice["priority"] == 1

    def leave_self_rate(votes):
        count = np.broadcast_to(votes.sum(axis=0),
                                (len(records), len(candidates))).copy()
        count[own_exists] -= votes[own[own_exists]]
        denominator = len(neighbors) - own_exists.astype(np.float32)
        return count / np.maximum(denominator[:, None], 1)

    global_rate = leave_self_rate(binary)
    global_priority = leave_self_rate(priority)
    global_first = leave_self_rate(first)
    semantic_votes = None
    if semantic_vectors is not None:
        assert semantic_vectors.shape[0] == len(modules)
        all_binary = np.zeros((len(neighbors), len(modules)), np.float32)
        all_priority = np.zeros_like(all_binary)
        all_first = np.zeros_like(all_binary)
        for i, choices in enumerate(previous):
            for choice in choices:
                j = index[choice["module_id"]]
                all_binary[i, j] = 1
                all_priority[i, j] = PRIORITY_WEIGHT[choice["priority"]]
                all_first[i, j] = choice["priority"] == 1
        ci = np.array([index[mid] for mid in candidates])
        similarity_courses = (semantic_vectors[ci] @ semantic_vectors.T).astype(np.float32)
        observed = all_binary.sum(axis=0) > 0
        allowed = np.broadcast_to(observed, similarity_courses.shape).copy()
        # Не переносим собственную популярность курса обратно на него же.
        allowed[np.arange(len(candidates)), ci] = False
        masked = np.where(allowed, similarity_courses, -2)
        kernel = np.exp(np.clip((masked - masked.max(axis=1, keepdims=True)) / .06,
                                -40, 0))
        kernel[~allowed] = 0
        kernel /= np.maximum(kernel.sum(axis=1, keepdims=True), 1e-8)
        assert np.all(kernel[np.arange(len(candidates)), ci] == 0)
        assert np.all(kernel[:, ~observed] == 0)
        semantic_votes = tuple(votes @ kernel.T
                               for votes in (all_binary, all_priority, all_first))
        semantic_global = tuple(leave_self_rate(votes) for votes in semantic_votes)
    # Стабильная сортировка фиксирует порядок соседей при одинаковом сходстве.
    order = np.argsort(-similarity, axis=1, kind="stable")[:, :max(NEIGHBOR_COUNTS)]
    n, k = len(records), len(candidates)
    active_names = HISTORY_NEIGHBOR_NAMES if semantic_vectors is None else HISTORY_NEIGHBOR_NAMES + HISTORY_SEMANTIC_NAMES
    output = np.zeros((n, k, len(active_names)), np.float32)
    diagnostics = {"source_students": len(neighbors),
                   "query_students": n,
                   "self_students_excluded": int(own_exists.sum()),
                   "queries_with_positive_neighbor": int((similarity.max(axis=1) > 0).sum()),
                   "candidate_seen_recent": (binary.sum(axis=0) > 0).tolist()}

    for block, count in enumerate(NEIGHBOR_COUNTS):
        selected = order[:, :count]
        sim = np.take_along_axis(similarity, selected, axis=1)
        weights = sim ** 2
        mass = weights.sum(axis=1)
        effective = mass ** 2 / np.maximum((weights ** 2).sum(axis=1), 1e-8)
        row = np.repeat(np.arange(n), selected.shape[1])
        weight_matrix = csr_matrix((weights.ravel(), (row, selected.ravel())),
                                   shape=(n, len(neighbors)))

        def smoothed(votes, global_values):
            raw = (weight_matrix @ votes) / np.maximum(mass[:, None], 1e-8)
            # Сглаживаем фиксированным эквивалентом десяти эффективных соседей.
            return ((raw * effective[:, None] + 10 * global_values)
                    / (effective[:, None] + 10))

        rate = smoothed(binary, global_rate)
        weighted_rate = smoothed(priority, global_priority)
        first_rate = smoothed(first, global_first)
        offset = block * 6
        output[:, :, offset] = np.log1p(1000 * rate)
        output[:, :, offset + 1] = np.log((rate + .004) / (global_rate + .004))
        output[:, :, offset + 2] = np.log1p(1000 * weighted_rate)
        output[:, :, offset + 3] = np.log1p(1000 * first_rate)
        output[:, :, offset + 4] = np.log1p(effective[:, None])
        mean_similarity = (weights * sim).sum(axis=1) / np.maximum(mass, 1e-8)
        output[:, :, offset + 5] = mean_similarity[:, None]
        diagnostics[f"top{count}_median_effective_neighbors"] = float(np.median(effective))
        diagnostics[f"top{count}_median_similarity"] = float(np.median(mean_similarity))
        if semantic_votes is not None:
            transferred = [smoothed(votes, global_values)
                           for votes, global_values in zip(semantic_votes, semantic_global)]
            offset = len(HISTORY_NEIGHBOR_NAMES) + block * 4
            output[:, :, offset] = np.log1p(1000 * transferred[0])
            output[:, :, offset + 1] = np.log(
                (transferred[0] + .004) / (semantic_global[0] + .004))
            output[:, :, offset + 2] = np.log1p(1000 * transferred[1])
            output[:, :, offset + 3] = np.log1p(1000 * transferred[2])

    assert np.isfinite(output).all()
    if semantic_votes is not None:
        diagnostics["semantic_donor_courses"] = int(observed.sum())
        diagnostics["semantic_kernel_self_weight_max"] = float(
            kernel[np.arange(len(candidates)), ci].max())
    return output, active_names, diagnostics


def full_feature_matrix(records, period, source, neighbor_source,
                        modules, similarities, vectors):
    """Объединяет три блока в точном порядке исходного эксперимента.

    0:35   — базовые, поведенческие и peer-признаки;
    35:43  — восемь семантических E5-признаков;
    43:96  — 53 признака тем, языков и интересов;
    96:116 — 20 обычных и семантических признаков соседей.
    """
    candidates, _, flat, base_names = feature_matrix(
        records, period, source, modules, similarities, vectors)
    base = flat.reshape(len(records), len(candidates), -1).astype(np.float32)
    interest, interest_names = interest_features(records, period, candidates, modules)
    neighbors, neighbor_names, _ = history_neighbor_features(
        records, period, neighbor_source, candidates, modules, vectors)
    names = base_names + interest_names + list(neighbor_names)
    matrix = np.concatenate((base, interest, neighbors), axis=2)
    assert matrix.shape == (len(records), len(candidates), 116)
    assert len(names) == len(set(names)) == 116
    assert np.isfinite(matrix).all()
    print(f"P{period}: {len(records)} студентов, {len(candidates)} курсов, "
          f"{len(names)} признаков", flush=True)
    return np.array(candidates), matrix, names

# ============================================================================
# 7. Обучаемая E5-совместимость
# ============================================================================

COMPATIBILITY_SETTINGS = {
    "dimensions": 64,
    "epochs": 80,
    "batch_size": 256,
    "learning_rate": 0.002,
    "weight_decay": 0.05,
    "dropout": 0.10,
    "temperature": 10.0,
    "recency": 0.7,
    "P3_weight": 1.5,
    "threads": 2,
}


COMPATIBILITY_SEED = 42004


COMPATIBILITY_SHARE = 0.25


def center_e5_vectors(vectors):
    """Центрируем и нормируем все1024 координаты, без PCA/SVD."""
    vectors = np.asarray(vectors, dtype=np.float32)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    centered /= np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    return centered.astype(np.float32)


def make_e5_history(records, period, module_ids, vectors):
    """Вход студента состоит только из выборов до целевого периода."""
    lookup = {mid: i for i, mid in enumerate(module_ids)}
    events = []
    for record in records:
        own = [
            (lookup[choice["module_id"]],
             PRIORITY_WEIGHT[choice["priority"]]
             * COMPATIBILITY_SETTINGS["recency"] ** (period - past["period"] - 1))
            for past in sorted(record["history"], key=lambda item: item["period"])
            if past["period"] < period
            for choice in sorted(past["choices"], key=lambda item: item["priority"])
        ]
        events.append(own)

    width = max(1, max(map(len, events), default=0))
    course = np.zeros((len(records), width), dtype=np.int64)
    weights = np.zeros((len(records), width), dtype=np.float32)
    for row, own in enumerate(events):
        for column, (index, weight) in enumerate(own):
            course[row, column] = index
            weights[row, column] = weight

    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    mean = (vectors[course] * weights[:, :, None]).sum(axis=1)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-8)
    return {
        "mean": torch.from_numpy(mean),
        "course": torch.from_numpy(course),
        "weights": torch.from_numpy(weights),
        "warm": weights.sum(axis=1) > 0,
    }


class E5Compatibility(nn.Module):
    """Две линейные проекции: история студента и курс-кандидат."""

    def __init__(self, vectors):
        super().__init__()
        self.register_buffer("vectors", torch.from_numpy(vectors))
        self.history = nn.Linear(1024, COMPATIBILITY_SETTINGS["dimensions"], bias=False)
        self.course = nn.Linear(1024, COMPATIBILITY_SETTINGS["dimensions"], bias=False)
        nn.init.orthogonal_(self.history.weight)
        self.course.weight.data.copy_(self.history.weight.data)
        self.dropout = nn.Dropout(COMPATIBILITY_SETTINGS["dropout"])
        # Влияние повторного выбора учится; жёсткого запрета повторов нет.
        self.repeat = nn.Parameter(torch.tensor(-2.))

    def forward(self, data, rows, candidates):
        history = F.normalize(self.history(self.dropout(data["mean"][rows])), dim=1)
        course = F.normalize(self.course(self.dropout(self.vectors[candidates])), dim=1)
        similarity = history @ course.T
        previous = data["course"][rows]
        weights = data["weights"][rows]
        repeated = (
            (previous[:, :, None] == candidates[None, None, :])
            & (weights[:, :, None] > 0)
        ).any(dim=1)
        return COMPATIBILITY_SETTINGS["temperature"] * similarity + self.repeat * repeated


def predict_e5_compatibility(model, data, candidates):
    """Обрабатываем P4 небольшими батчами в режиме без dropout."""
    model.eval()
    predictions = []
    size = COMPATIBILITY_SETTINGS["batch_size"]
    with torch.no_grad():
        for start in range(0, len(data["mean"]), size):
            rows = np.arange(start, min(start + size, len(data["mean"])))
            predictions.append(model(data, rows, candidates).numpy())
    return np.vstack(predictions).astype(np.float32)


def train_e5_compatibility(vectors, datasets, query, query_candidates, seed=42004):
    """Обучаемся на доступных каталогах P2/P3; ответы P4 не нужны.

    На студента приходится несколько положительных курсов. Приоритеты
    задают целевое распределение внутри доступного каталога.
    Холодные студенты не обучают представление истории.
    """
    torch.set_num_threads(COMPATIBILITY_SETTINGS["threads"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = E5Compatibility(vectors)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=COMPATIBILITY_SETTINGS["learning_rate"],
        weight_decay=COMPATIBILITY_SETTINGS["weight_decay"],
    )

    prepared = []
    for data, rows, candidates, gains, period_weight in datasets:
        rows = np.asarray(rows)
        rows = rows[data["warm"][rows] & (gains[rows].sum(axis=1) > 0)]
        target = gains[rows].copy()
        target /= target.sum(axis=1, keepdims=True)
        prepared.append((data, rows, candidates, torch.from_numpy(target), period_weight))

    schedule = np.asarray([
        (part, row)
        for part, item in enumerate(prepared)
        for row in range(len(item[1]))
    ], dtype=np.int64)
    size = COMPATIBILITY_SETTINGS["batch_size"]
    final_prediction = None

    for epoch in range(1, COMPATIBILITY_SETTINGS["epochs"] + 1):
        model.train()
        numerator = denominator = 0.
        permutation = rng.permutation(len(schedule))
        for start in range(0, len(schedule), size):
            batch = schedule[permutation[start:start + size]]
            optimizer.zero_grad(set_to_none=True)
            loss = total_weight = 0.
            for part, (data, rows, candidates, target, weight) in enumerate(prepared):
                own = batch[batch[:, 0] == part, 1]
                if not len(own):
                    continue
                logits = model(data, rows[own], candidates)
                per_query = -(target[own] * F.log_softmax(logits, dim=1)).sum(dim=1)
                loss = loss + weight * per_query.sum()
                total_weight += weight * len(own)
            loss = loss / total_weight
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            numerator += float(loss.detach()) * total_weight
            denominator += total_weight

        if epoch % 10 == 0:
            print(f"E5-совместимость: эпоха{epoch}, loss={numerator / denominator:.6f}", flush=True)
        # Сохраняем тот же режим переключения train/eval, что в исходной модели.
        # Предсказания40-й эпохи не записываем на диск и в сабмите не используем.
        if epoch in (40, 80):
            final_prediction = predict_e5_compatibility(model, query, query_candidates)

    del model, optimizer, prepared
    gc.collect()
    return final_prediction

E5_FEATURE_NAMES = [
    'e5_learned_compatibility', 'e5_learned_student_percentile',
    'e5_frozen_compatibility', 'e5_learned_minus_frozen',
]


def subset_history(history, rows):
    return {name: value[rows] for name, value in history.items()}


def frozen_e5_scores(history, vectors, candidates):
    score = history['mean'].numpy() @ vectors[candidates].T
    previous = history['course'].numpy()
    weights = history['weights'].numpy()
    repeated = ((previous[:, :, None] == candidates[None, None, :])
                & (weights[:, :, None] > 0)).any(axis=1)
    return (10.0 * score - 2 * repeated).astype(np.float32)


def e5_tree_features(raw, history, vectors, candidates):
    frozen = frozen_e5_scores(history, vectors, candidates)
    ranks = np.argsort(np.argsort(raw, axis=1, kind='stable'), axis=1, kind='stable')
    percentile = ranks.astype(np.float32) / max(raw.shape[1] - 1, 1)
    values = np.stack((raw / 10, percentile, frozen / 10, (raw - frozen) / 10), axis=2)
    values[~history['warm']] = 0
    assert np.isfinite(values).all()
    return values.astype(np.float32)


def build_e5_predictions(p2, val, test, c2, c3, c4, y2, y3, modules, original_vectors):
    vectors = center_e5_vectors(original_vectors)
    module_ids = [module['module_id'] for module in modules]
    lookup = {mid: i for i, mid in enumerate(module_ids)}
    ci2, ci3, ci4 = [np.array([lookup[mid] for mid in c]) for c in (c2, c3, c4)]
    h2 = make_e5_history(p2, 2, module_ids, vectors)
    h3 = make_e5_history(val, 3, module_ids, vectors)
    h4 = make_e5_history(test, 4, module_ids, vectors)
    all2 = np.arange(len(p2))

    raw2 = np.zeros_like(y2)
    history2 = np.array([len(r['history']) for r in p2])
    folds2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=52002)
    for fold, (fit, hold) in enumerate(folds2.split(all2, history2)):
        assert not np.intersect1d(fit, hold).size
        print(f'E5 P2 crossfit: {fold + 1}/3', flush=True)
        datasets = [(subset_history(h2, fit), np.arange(len(fit)), ci2, y2[fit].copy(), 1.0)]
        raw2[hold] = train_e5_compatibility(vectors, datasets, subset_history(h2, hold), ci2, seed=52000 + fold)
    extra2 = e5_tree_features(raw2, h2, vectors, ci2)

    extra3 = np.zeros((*y3.shape, 4), np.float32)
    history3 = np.array([len(r['history']) for r in val])
    folds3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=54003)
    for fold, (fit, hold) in enumerate(folds3.split(np.arange(len(val)), history3)):
        assert not np.intersect1d(fit, hold).size
        print(f'E5 P3 crossfit: {fold + 1}/3', flush=True)
        datasets = [(h2, all2, ci2, y2, 1.0),
                    (subset_history(h3, fit), np.arange(len(fit)), ci3, y3[fit].copy(), 1.5)]
        query = subset_history(h3, hold)
        raw = train_e5_compatibility(vectors, datasets, query, ci3, seed=54000 + fold)
        extra3[hold] = e5_tree_features(raw, query, vectors, ci3)

    print('E5: финальная модель P2/P3 → P4', flush=True)
    datasets = [(h2, all2, ci2, y2, 1.0),
                (h3, np.arange(len(val)), ci3, y3, 1.5)]
    raw4 = train_e5_compatibility(vectors, datasets, h4, ci4, seed=42004)
    extra4 = e5_tree_features(raw4, h4, vectors, ci4)
    return extra2, extra3, extra4, raw4, vectors

# ============================================================================
# 8. Ridge по точным курсам истории с E5-ядром
# ============================================================================

def ridge_vocabulary(modules, records):
    return {"modules": [m["module_id"] for m in modules],
            "institute": sorted({r["profile"]["institute"] for r in records if r["profile"]["institute"]}),
            "specialty": sorted({r["profile"]["specialty"] for r in records if r["profile"]["specialty"]})}


def ridge_history_features(records, period, vocab):
    index = {mid: j for j, mid in enumerate(vocab["modules"])}
    width = len(index)
    output = np.zeros((len(records), 2 * width + len(vocab["institute"]) + len(vocab["specialty"])), np.float64)
    for row, record in enumerate(records):
        for history in record["history"]:
            if history["period"] >= period:
                continue
            block = 0 if history["period"] == period - 1 else width
            decay = .7 ** (period - 1 - history["period"])
            for choice in history["choices"]:
                output[row, block + index[choice["module_id"]]] += decay * PRIORITY_WEIGHT[choice["priority"]]
    # Нормируем блоки независимо: длина истории не увеличивает масштаб.
    for start in (0, width):
        own = output[:, start:start + width]
        own /= np.maximum(np.linalg.norm(own, axis=1, keepdims=True), 1e-8)
    offset = 2 * width
    for key in ("institute", "specialty"):
        lookup = {value: j for j, value in enumerate(vocab[key])}
        for row, record in enumerate(records):
            value = record["profile"][key]
            if value in lookup:
                output[row, offset + lookup[value]] = .5
        offset += len(lookup)
    assert np.isfinite(output).all()
    return output


def exact_ridge_fit(x2, y2, c2, x3, y3, c3, fit_rows, hold, columns, alpha, target):
    """Один solve на группу курсов с одинаковыми периодами доступности.

Все P2 ответы допустимы, а P3 ответы — только fit_rows и columns.
Проверочная y3 нигде не используется. Интерсепт не регуляризуется.
"""
    assert target in ("hit", "gain") and alpha > 0
    assert not np.intersect1d(fit_rows, hold).size
    p2_index = {mid: j for j, mid in enumerate(c2)}
    groups = {}
    for column in columns:
        signature = (str(c3[column]) in p2_index, bool(len(fit_rows)))
        groups.setdefault(signature, []).append(column)
    raw = np.zeros((len(hold), len(c3)), np.float32)
    coefficients = np.zeros((x3.shape[1], len(c3)), np.float64)
    intercepts = np.zeros(len(c3), np.float64)
    receipt = []
    for (has_p2, has_p3), own in groups.items():
        if not has_p2 and not has_p3:
            continue
        inputs, labels, weights = [], [], []
        if has_p2:
            inputs.append(x2)
            labels.append(y2[:, [p2_index[str(c3[c])] for c in own]])
            weights.append(np.ones(len(x2)))
        if has_p3:
            inputs.append(x3[fit_rows])
            labels.append(y3[np.ix_(fit_rows, own)])
            weights.append(np.full(len(fit_rows), 1.5))
        x, y, weight = np.vstack(inputs), np.vstack(labels), np.concatenate(weights)
        if target == "hit":
            y = (y > 0).astype(np.float64)
        total = weight.sum()
        mx, my = weight @ x / total, weight @ y / total
        centered_x, centered_y = x - mx, y - my
        gram = centered_x.T @ (centered_x * weight[:, None])
        gram.flat[::len(gram) + 1] += alpha
        rhs = centered_x.T @ (centered_y * weight[:, None])
        coef = cho_solve(cho_factor(gram, lower=True, check_finite=False), rhs, check_finite=False)
        bias = my - mx @ coef
        raw[:, own] = (x3[hold] @ coef + bias).astype(np.float32)
        coefficients[:, own], intercepts[own] = coef, bias
        receipt.append({"courses": [str(c3[c]) for c in own], "P2_queries": len(x2) if has_p2 else 0,
                        "P3_queries": len(fit_rows) if has_p3 else 0,
                        "negative_coefficients": int((coef < 0).sum()), "alpha": alpha, "target": target})
    assert np.isfinite(raw).all()
    return raw, coefficients, intercepts, receipt

RIDGE_FEATURE_NAMES = ['joint_history_gain', 'joint_history_student_percentile']


def ridge_tree_features(raw, warm):
    ranks = np.argsort(np.argsort(raw, axis=1, kind='stable'), axis=1, kind='stable')
    output = np.stack((raw, ranks / max(raw.shape[1] - 1, 1)), axis=2).astype(np.float32)
    output[~warm] = 0
    return output


def semantic_ridge_fit(x2, y2, c2, x3, y3, c3, fit, hold, columns, factor):
    width = len(factor)
    transformed = []
    for original in (x2, x3):
        changed = original.copy()
        for start in (0, width):
            changed[:, start:start + width] = original[:, start:start + width] @ factor
        transformed.append(changed)
    return exact_ridge_fit(transformed[0], y2, c2, transformed[1], y3, c3,
                           fit, hold, columns, 10.0, 'gain')[0]


def build_ridge_predictions(train, p2, val, test, c2, c3, c4, y2, y3, modules, vectors, new_courses):
    vocab = ridge_vocabulary(modules, train + val + test)
    x2 = ridge_history_features(p2, 2, vocab)
    x3 = ridge_history_features(val, 3, vocab)
    x4 = ridge_history_features(test, 4, vocab)
    factor = np.linalg.cholesky((np.eye(len(vectors)) + .5 * (vectors.astype(np.float64) @ vectors.astype(np.float64).T)) / 1.5)
    extra2 = np.zeros((*y2.shape, 2), np.float32)
    splitter2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=57002)
    for fold, (fit, hold) in enumerate(splitter2.split(np.arange(len(p2)), [len(r['history']) for r in p2])):
        print(f'Ridge P2 crossfit: {fold + 1}/3', flush=True)
        raw = semantic_ridge_fit(x2[fit], y2[fit], c2, x2[hold],
            np.zeros((len(hold), len(c2)), np.float32), c2, np.array([], int),
            np.arange(len(hold)), np.arange(len(c2)), factor)
        extra2[hold] = ridge_tree_features(raw, np.array([bool(p2[i]['history']) for i in hold]))
    extra3 = np.zeros((*y3.shape, 2), np.float32)
    splitter3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=57004)
    for fold, (fit, hold) in enumerate(splitter3.split(np.arange(len(val)), [len(r['history']) for r in val])):
        assert not np.intersect1d(fit, hold).size
        print(f'Ridge P3 crossfit: {fold + 1}/3', flush=True)
        raw = semantic_ridge_fit(x2, y2, c2, x3, y3, c3, fit, hold, np.arange(len(c3)), factor)
        extra3[hold] = ridge_tree_features(raw, np.array([bool(val[i]['history']) for i in hold]))
    raw4 = np.zeros((len(test), len(c4)), np.float32)
    index3 = {mid: j for j, mid in enumerate(c3)}
    for available3 in (False, True):
        own = np.array([j for j, mid in enumerate(c4) if not new_courses[j] and (mid in index3) == available3], int)
        if not len(own): continue
        courses = c4[own]
        if available3:
            query = np.vstack((x3, x4))
            labels = np.vstack((y3[:, [index3[mid] for mid in courses]], np.zeros((len(x4), len(own)), np.float32)))
            fit = np.arange(len(x3)); hold = np.arange(len(x3), len(query))
        else:
            query = x4; labels = np.zeros((len(x4), len(own)), np.float32)
            fit = np.array([], int); hold = np.arange(len(x4))
        raw4[:, own] = semantic_ridge_fit(x2, y2, c2, query, labels, courses, fit, hold, np.arange(len(own)), factor)
    extra4 = ridge_tree_features(raw4, np.array([bool(r['history']) for r in test]))
    return extra2, extra3, extra4, raw4

# ============================================================================
# 9. LightGBM и CatBoost
# ============================================================================

def training_target(y2, y3, use_priorities):
    """Бинарная метка и веса: P3 важнее P2, приоритет влияет на positives.

    Значения y2/y3 — веса официальных приоритетов, а не вероятности.
    Отрицательные примеры получают только вес своего периода.
    """
    gains = np.r_[y2.ravel(), y3.ravel()]
    target = (gains > 0).astype(np.int8)
    weight = np.r_[np.ones(y2.size), np.full(y3.size, P3_TRAIN_WEIGHT)]
    if use_priorities:
        weight *= np.where(gains > 0, gains / PRIORITY_SCALE, 1)
    return target, weight


def category_matrix(records, candidates):
    """Категории всех пар идут в том же порядке, что и численные признаки.

    FeaturesData ожидает строковые категории в виде UTF-8 bytes.
    ID студента в модели не используется.
    """
    k = len(candidates)
    values = np.empty((len(records) * k, len(CATEGORICAL_COLUMNS)), dtype=object)
    values[:, 0] = np.tile(
        np.array([mid.encode("utf-8") for mid in candidates], dtype=object), len(records))
    for j, key in enumerate(CATEGORICAL_COLUMNS[1:], start=1):
        categories = np.array(
            [(record["profile"][key] or "_missing_").encode("utf-8")
             for record in records], dtype=object)
        values[:, j] = np.repeat(categories, k)
    return values.reshape(len(records), k, len(CATEGORICAL_COLUMNS))

TREE_WORKER_CODE = r'''
"""Отдельный процесс деревьев: PyTorch сюда не импортируется.

Основной скрипт передаёт временные массивы и JSON с фиксированными
параметрами. Это устраняет конфликт OpenMP PyTorch/LightGBM на macOS.
Категориальные массивы создаются основным процессом решения.
"""

import json
import sys
import time

import numpy as np


def main(directory):
    from pathlib import Path

    started = time.monotonic()
    directory = Path(directory)
    config = json.loads((directory/"config.json").read_text())
    numeric = np.load(directory/"numeric.npy", mmap_mode="r")
    target = np.load(directory/"target.npy", mmap_mode="r")
    labels = np.load(directory/"labels.npy")
    weights = np.load(directory/"weights.npy")
    names = config["names"]
    if config["key"] == "catboost":
        from catboost import CatBoostClassifier, FeaturesData, Pool

        categorical = np.load(directory/"categorical.npy", allow_pickle=True)
        target_categorical = np.load(directory/"target_categorical.npy", allow_pickle=True)

        def pool(numeric, categorical, labels=None, weights=None):
            data = FeaturesData(num_feature_data=np.ascontiguousarray(numeric, dtype=np.float32),
                                cat_feature_data=np.ascontiguousarray(categorical, dtype=object),
                                num_feature_names=names,
                                cat_feature_names=["module_id", "institute", "specialty"])
            return Pool(data, label=labels, weight=weights)

        model = CatBoostClassifier(**config["parameters"])
        model.fit(pool(numeric, categorical, labels, weights), verbose=200)
        assert model.tree_count_ == config["parameters"]["iterations"]
        raw = model.predict_proba(pool(target, target_categorical))[:, 1]
        importance = model.get_feature_importance()
    else:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(**config["parameters"])
        model.fit(numeric, labels, sample_weight=weights, feature_name=names)
        assert model.booster_.num_trees() == config["parameters"]["n_estimators"]
        raw = model.predict_proba(target)[:, 1]
        importance = model.booster_.feature_importance(importance_type="gain")
    assert np.isfinite(raw).all()
    np.save(directory/"raw.npy", raw)
    (directory/"result.json").write_text(json.dumps({"importance": importance.tolist(),
                                                    "elapsed_seconds": time.monotonic()-started}))
    print("ISOLATED TREE COMPLETE", config["key"], round(time.monotonic()-started, 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])

'''

def fit_tree(label, key, x2, x3, x4, y2, y3, names, columns, parameters,
             use_priorities, cats2=None, cats3=None, cats4=None, float32_output=True):
    print(f'Обучение: {label}', flush=True)
    columns = np.asarray(columns)
    feature_names = [names[i] for i in columns]
    numeric = np.vstack((x2[:, :, columns].reshape(-1, len(columns)),
                         x3[:, :, columns].reshape(-1, len(columns))))
    target = x4[:, :, columns].reshape(-1, len(columns))
    labels, weights = training_target(y2, y3, use_priorities)
    with tempfile.TemporaryDirectory(prefix='mcu_tree_') as directory:
        for name, value in [('numeric', numeric), ('target', target), ('labels', labels), ('weights', weights)]:
            np.save(os.path.join(directory, name + '.npy'), value)
        if key == 'catboost':
            np.save(os.path.join(directory, 'categorical.npy'), np.vstack((cats2.reshape(-1, 3), cats3.reshape(-1, 3))))
            np.save(os.path.join(directory, 'target_categorical.npy'), cats4.reshape(-1, 3))
        with open(os.path.join(directory, 'config.json'), 'w', encoding='utf-8') as file:
            json.dump({'key': key, 'names': feature_names, 'parameters': parameters}, file)
        worker_path = os.path.join(directory, 'tree_worker.py')
        with open(worker_path, 'w', encoding='utf-8') as file:
            file.write(TREE_WORKER_CODE)
        del numeric, target, labels, weights
        gc.collect()
        subprocess.run([sys.executable, worker_path, directory], check=True)
        prediction = np.load(os.path.join(directory, 'raw.npy')).reshape(x4.shape[:2])
    if float32_output: prediction = prediction.astype(np.float32)
    assert np.isfinite(prediction).all()
    return prediction

# ============================================================================
# 10. Смесь прогнозов, новые курсы и холодный порядок
# ============================================================================

def stable_rank(scores):
    """Ранги 0..1; одинаковые оценки разрешаем по порядку кандидатов.

    Эта функция повторяет ранжирование CatBoost-смеси в эксперименте.
    percentile_rank выше сохранена для прежнего бонуса избранного и
    прежней холодной смеси: их обработка равных значений отличается.
    """
    descending = np.argsort(-scores, axis=1, kind="stable")
    positions = np.argsort(descending, axis=1, kind="stable")
    return ((scores.shape[1] - 1 - positions) / (scores.shape[1] - 1)).astype(np.float32)


def combine_lightgbm(old_full, old_content, old_priority,
                     large_peer, large_priority, new_courses):
    """Точный old/new ансамбль; у впервые появившихся курсов только peer."""
    old_peer = (old_full + old_content) / 2
    old = (1 - PRIORITY_SHARE) * old_peer + PRIORITY_SHARE * old_priority
    old[:, new_courses] = old_peer[:, new_courses]
    large = (1 - PRIORITY_SHARE) * large_peer + PRIORITY_SHARE * large_priority
    large[:, new_courses] = large_peer[:, new_courses]
    return (1 - NEW_ENSEMBLE_SHARE) * old + NEW_ENSEMBLE_SHARE * large


def adjust_new_course_prior(raw, new_courses, warm, signal_counts, settings):
    """Перераспределить исходную среднюю оценку внутри новых курсов.

    1. Получаем относительные доли исходной модели и внешнего сигнала.
    2. Смешиваем их в логарифмической шкале.
    3. Сохраняем суммарную среднюю оценку всех новых курсов до ограничения.
    4. Подбираем один сдвиг на курс, одинаковый для всех тёплых студентов.
    5. Ограничиваем сдвиг: исходные индивидуальные различия сохраняются.

    Старые курсы и холодные строки остаются побитово прежними.
    Ограничение сдвигов может немного изменить итоговую сумму средних.
    Эта сумма не является квотой показов или числом будущих выборов.
    """
    raw = np.asarray(raw)
    new_courses = np.asarray(new_courses, dtype=bool)
    warm = np.asarray(warm, dtype=bool)
    counts = np.asarray(signal_counts, dtype=float)
    assert raw.ndim == 2 and raw.shape == (len(warm), len(new_courses))
    assert counts.shape == new_courses.shape and np.isfinite(counts).all()
    assert (counts >= 0).all() and np.isfinite(raw).all()
    assert warm.any() and new_courses.any()
    assert 0 <= settings["strength"] <= 1 and settings["max_odds_factor"] >= 1
    output = raw.copy()
    selected = np.ix_(warm, new_courses)
    q = np.clip(raw[selected].astype(float), 1e-6, 1-1e-6)
    mean = q.mean(axis=0)
    model_share = mean / mean.sum()
    signal_share = counts[new_courses] + settings["pseudocount"]
    signal_share /= signal_share.sum()
    mixed = np.exp((1-settings["strength"])*np.log(model_share)
                   + settings["strength"]*np.log(signal_share))
    target = mean.sum() * mixed / mixed.sum()
    # В данном каталоге значения малы, но не допускаем недостижимую цель.
    target = np.clip(target, 1e-6, 1-1e-6)
    z = logit(q)
    lo, hi = np.full(len(mean), -20.), np.full(len(mean), 20.)
    for _ in range(45):
        middle = (lo+hi)/2
        predicted_mean = expit(z+middle).mean(axis=0)
        lo = np.where(predicted_mean < target, middle, lo)
        hi = np.where(predicted_mean >= target, middle, hi)
    unconstrained = (lo+hi)/2
    bound = np.log(settings["max_odds_factor"])
    offset = np.clip(unconstrained, -bound, bound)
    if settings["strength"]:
        output[selected] = expit(z+offset).astype(raw.dtype)
    assert np.array_equal(output[:, ~new_courses], raw[:, ~new_courses])
    assert np.array_equal(output[~warm], raw[~warm])
    details = {
        "model_share": model_share, "signal_share": signal_share,
        "target_mean": target, "original_mean": mean,
        "adjusted_mean": output[selected].mean(axis=0),
        "offset": offset, "unconstrained_offset": unconstrained,
    }
    return output, details

def match_blend(raw, expert, warm, folds, share, columns):
    """По каждому курсу сохраняем исходный набор оценок перед смешиванием."""
    output = raw.copy()
    if share == 0: return output
    for fold in np.unique(folds):
        rows = np.flatnonzero((folds == fold) & warm)
        if not len(rows): continue
        selected = np.ix_(rows, columns)
        q = raw[selected]; e = expert[selected]
        # Вторичный порядок базы делает ничьи воспроизводимыми.
        order = np.lexsort((q, e),axis=0)
        aligned = np.empty_like(q)
        aligned[order, np.arange(q.shape[1])] = np.sort(q,axis=0)
        assert np.array_equal(np.sort(aligned,axis=0),np.sort(q,axis=0))
        output[selected] = (1-share)*q + share*aligned
        assert np.allclose(output[selected].mean(axis=0),q.mean(axis=0),atol=1e-7)
    assert np.array_equal(output[~warm],raw[~warm])
    excluded = np.ones(raw.shape[1],bool); excluded[columns] = False
    assert np.array_equal(output[:,excluded],raw[:,excluded])
    return output

ANCHORS = 2
DISCOUNT = 1 / np.log2(np.arange(2, 7))

def best_order(gains, weights, anchors=ANCHORS):
    """Перебираем хвост пятёрки; первые два места сохраняем.

    Считаем средний min(1, DCG), поэтому учитываем совместные попадания
    и насыщение официальной метрики, а не только отдельную популярность.
    """
    prefix = tuple(range(anchors))
    best_value, selected = -1., None
    for tail in itertools.permutations(range(anchors, 5)):
        order = prefix + tail
        value = float(weights @ np.minimum(1., gains[:, order] @ DISCOUNT))
        if value > best_value + 1e-12:
            best_value, selected = value, order
    return np.array(selected), best_value

def match_favorite_marginals(gains, desired):
    """Мягко перевзвешиваем прошлых студентов под текущий спрос.

    IPF / raking: после обновления по курсу его взвешенная частота
    совпадает с целевой. Совместные выборы и приоритеты берём из P3.
    Целевые частоты по избранному являются эвристикой, а не ответами P4.
    """
    chosen = gains > 0
    assert np.all(chosen.any(axis=0)) and np.all((~chosen).any(axis=0))
    assert np.all((desired > 0) & (desired < 1))
    weights = np.full(len(gains), 1 / len(gains))
    for iteration in range(200):
        for j in range(5):
            mask = chosen[:, j]
            current = float(weights @ mask)
            weights[mask] *= desired[j] / current
            weights[~mask] *= (1 - desired[j]) / (1 - current)
        weights /= weights.sum()
        if np.max(np.abs(weights @ chosen - desired)) < 1e-10:
            break
    assert np.max(np.abs(weights @ chosen - desired)) < 1e-8
    return weights

def optimized_cold_order(candidates, favorites, past, y3, c3):
    previous = Counter(choice['module_id'] for r in past for h in r['history']
                       if h['period'] == 3 for choice in h['choices'])
    recent = np.array([previous[mid] for mid in candidates], np.float32)
    favorite_rank = np.argsort(np.argsort(favorites)).astype(np.float32) / len(candidates)
    recent_rank = np.argsort(np.argsort(recent)).astype(np.float32) / len(candidates)
    before = np.argsort(-(.8 * favorite_rank + .2 * recent_rank), kind='stable')[:5]
    index3 = {mid: j for j, mid in enumerate(c3)}
    gains = np.zeros((len(y3), len(candidates)), np.float64)
    for j, mid in enumerate(candidates):
        if mid in index3: gains[:, j] = y3[:, index3[mid]]
    five = gains[:, before]
    favorite_probability = 5 * favorites.astype(float) / favorites.sum()
    previous_probability = (gains > 0).mean(axis=0)
    desired = (.8 * favorite_probability + .2 * previous_probability)[before]
    weights = match_favorite_marginals(five, desired)
    permutation, _ = best_order(five, weights)
    return before[permutation]


def finish_ensemble(old_mixed, legacy_raw, legacy_cat, peer, priority, e5_cat, ridge_cat,
                    e5_raw, ridge_raw, new_courses, test, candidates, modules, past, y3, c3):
    warm = np.array([bool(r['history']) for r in test])
    old = np.flatnonzero(~new_courses)
    folds = np.zeros(len(test), int)
    raw = .5 * old_mixed + .5 * (.25 * peer + .75 * priority)
    raw[:, new_courses], raw[~warm] = legacy_raw[:, new_courses], legacy_raw[~warm]
    old_cat = e5_cat.copy()
    old_cat[:, new_courses], old_cat[~warm] = legacy_cat[:, new_courses], legacy_cat[~warm]
    cat = ridge_cat.copy()
    cat[:, new_courses], cat[~warm] = old_cat[:, new_courses], old_cat[~warm]
    cat = match_blend(old_cat, cat, warm, folds, 1.0, old)
    adjusted = match_blend(raw, e5_raw, warm, folds, .25, old)
    modules_by_id = {m['module_id']: m for m in modules}
    favorites = np.array([int(FAVORITE_PATTERN.search(modules_by_id[mid]['description']).group(1))
                          for mid in candidates], np.float32)
    adjusted = adjust_new_course_prior(adjusted, new_courses, warm, favorites, NEW_COURSE_PRIOR)[0]
    adjusted = match_blend(adjusted, ridge_raw, warm, folds, .25, old)
    score = .75 * stable_rank(adjusted) + .25 * stable_rank(cat)
    ranked = stable_rank(score) + .05 * percentile_rank(np.broadcast_to(favorites, score.shape))
    order = np.argsort(-ranked, axis=1, kind='stable')[:, :5]
    cold_order = optimized_cold_order(candidates, favorites, past, y3, c3)
    order[~warm] = cold_order
    print('Холодная пятёрка:', ' '.join(candidates[cold_order]), flush=True)
    return order, ~warm

def write_submission(test, candidates, order, cold):
    """Проверяет ID и пятёрки, затем записывает единственный выходной CSV."""
    ids = [record["id"] for record in test]
    if len(ids) != len(set(ids)):
        raise ValueError("В test.jsonl найдены повторяющиеся ID")
    output = os.path.join(PROJECT_DIR, OUTPUT_NAME)
    with open(output, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "recommendations"])
        for record, positions in zip(test, order):
            selected = candidates[positions].tolist()
            assert len(selected) == len(set(selected)) == 5
            assert set(selected) <= set(record["available_modules"])
            writer.writerow([record["id"], " ".join(selected)])
    print(f"Сохранено: {output}", flush=True)
    print(f"Строк: {len(test)}, студентов без истории: {int(cold.sum())}", flush=True)

# ============================================================================
# 11. Полный запуск
# ============================================================================

def run_solution():
    required = ("train.jsonl", "val.jsonl", "test.jsonl", "modules.jsonl", "val_solution.csv")
    missing = [name for name in required if not os.path.isfile(os.path.join(DATA_DIR, name))]
    if missing:
        raise FileNotFoundError(
            "Положите исходные файлы соревнования в " + DATA_DIR
            + ". Не найдены: " + ", ".join(missing))
    print('[1/7] Чтение исходных данных', flush=True)
    train = read_jsonl('train.jsonl'); val = read_jsonl('val.jsonl')
    test = read_jsonl('test.jsonl'); modules = read_jsonl('modules.jsonl')
    answers3 = read_answers()
    p2, answers2_list = period2_examples(train, modules)
    answers2 = {r['id']: answer for r, answer in zip(p2, answers2_list)}
    past = complete_past(train, val, test, answers3)
    neighbor_source = merge_past_with_ids(train, val, test, answers3)

    print('[2/7] Тексты и 116 признаков', flush=True)
    similarities = text_similarity(modules)
    vectors = get_embeddings(modules)
    c2, x2, names = full_feature_matrix(p2, 2, train, train, modules, similarities, vectors)
    c3, x3, names3 = full_feature_matrix(val, 3, train, train, modules, similarities, vectors)
    c4, x4, names4 = full_feature_matrix(test, 4, past, neighbor_source, modules, similarities, vectors)
    assert names == names3 == names4
    assert all(r['available_modules'] == c4.tolist() for r in test)
    y2 = priority_labels(p2, c2, answers2).reshape(x2.shape[:2])
    y3 = priority_labels(val, c3, answers3).reshape(x3.shape[:2])
    module_by_id = {m['module_id']: m for m in modules}
    new = np.array([not any(p < 4 for p in module_by_id[mid]['available_in_periods']) for mid in c4])
    content = [i for i, name in enumerate(names[:35]) if not (name.startswith('peer_') and '_title_' in name)]
    peer_columns = np.r_[np.arange(35), np.arange(43, 116)]
    cats2, cats3, cats4 = [category_matrix(records, candidates) for records, candidates in ((p2, c2), (val, c3), (test, c4))]

    print('[3/7] Исходные модели для смеси и новых курсов', flush=True)
    def lgb_fit(label, columns, parameters, priority, f32=False):
        return fit_tree(label, 'lightgbm', x2, x3, x4, y2, y3, names, columns,
                        parameters, priority, float32_output=f32)
    old_full = lgb_fit('old peer full', np.arange(35), BASE_LGB_PARAMS, False)
    old_content = lgb_fit('old peer content', content, BASE_LGB_PARAMS, False)
    old_priority = lgb_fit('old priority', np.arange(43), {**BASE_LGB_PARAMS, 'random_state': 42}, True)
    old_peer = (old_full + old_content) / 2
    old_mixed = .25 * old_peer + .75 * old_priority
    old_mixed[:, new] = old_peer[:, new]
    del old_full, old_content, old_priority, old_peer
    large_peer = lgb_fit('legacy large peer', peer_columns, LARGE_LGB_PARAMS, False, True)
    large_priority = lgb_fit('legacy large priority', np.arange(116), LARGE_LGB_PARAMS, True, True)
    large = .25 * large_peer + .75 * large_priority
    large[:, new] = large_peer[:, new]
    legacy_raw = .5 * old_mixed + .5 * large
    del large, large_peer, large_priority
    legacy_cat = fit_tree('legacy CatBoost', 'catboost', x2, x3, x4, y2, y3, names,
        np.arange(116), CATBOOST_PARAMS, True, cats2, cats3, cats4)

    print('[4/7] E5-учителя и crossfit-признаки', flush=True)
    e2, e3, e4, e5_raw, centered_vectors = build_e5_predictions(p2, val, test, c2, c3, c4, y2, y3, modules, vectors)
    print('[5/7] Ridge-учителя и crossfit-признаки', flush=True)
    r2, r3, r4, ridge_raw = build_ridge_predictions(train, p2, val, test, c2, c3, c4, y2, y3, modules, centered_vectors, new)

    print('[6/7] Деревья с обучаемыми признаками', flush=True)
    def stacked_tree(label, key, columns, priority, ridge=False):
        feature_names = [names[i] for i in columns] + E5_FEATURE_NAMES
        additions = [(e2, r2), (e3, r3), (e4, r4)]
        matrices = []
        for numeric, (e, r) in zip((x2, x3, x4), additions):
            pieces = [numeric[:, :, columns], e]
            if ridge: pieces.append(r)
            matrices.append(np.concatenate(pieces, axis=2))
        if ridge: feature_names += RIDGE_FEATURE_NAMES
        params = CATBOOST_PARAMS if key == 'catboost' else LARGE_LGB_PARAMS
        result = fit_tree(label, key, *matrices, y2, y3, feature_names,
            np.arange(len(feature_names)), params, priority, cats2, cats3, cats4)
        del matrices; gc.collect()
        return result
    peer = stacked_tree('E5 peer LightGBM', 'lightgbm', peer_columns, False)
    priority = stacked_tree('E5 priority LightGBM', 'lightgbm', np.arange(116), True)
    e5_cat = stacked_tree('E5 CatBoost', 'catboost', np.arange(116), True)
    ridge_cat = stacked_tree('E5 + Ridge CatBoost', 'catboost', np.arange(116), True, ridge=True)
    del x2, x3, x4, e2, e3, e4, r2, r3, r4
    gc.collect()

    print('[7/7] Финальная смесь и сохранение CSV', flush=True)
    order, cold = finish_ensemble(old_mixed, legacy_raw, legacy_cat, peer, priority,
        e5_cat, ridge_cat, e5_raw, ridge_raw, new, test, c4, modules, past, y3, c3)
    write_submission(test, c4, order, cold)
    return order

if __name__ == "__main__":
    run_solution()
