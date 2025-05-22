import numpy as np
import networkx as nx
from typing import Tuple


def leblanc_algorithm(T_0: np.ndarray, D: np.ndarray, C: np.ndarray, epsilon: float) -> Tuple[np.ndarray, int]:
    """
    Args:
        T_0: Начальная матрица затрат
        D: Матрица спроса
        C: Матрица пропускных способностей
        epsilon: Критерий остановки (максимальное относительное изменение)

    Returns:
        Кортеж (матрица потоков, число итераций)
    """
    # print('leblanc algorithm started')
    n = len(D)
    graph = nx.from_numpy_array(T_0, create_using=nx.DiGraph)
    X = get_flow_matrix(graph, n, D)
    iteration = 1

    while True:
        # Шаг 1: Обновление матрицы затрат
        T = T_0 * (1 + 0.15 * (X / C)**4)
        np.nan_to_num(T, copy=False, posinf=0.0, neginf=0.0)

        # Шаг 2: Расчет нового потока
        new_graph = nx.from_numpy_array(T, create_using=nx.DiGraph)
        Y = get_flow_matrix(new_graph, n, D)

        # Шаг 3: Поиск оптимального lambda
        C_inv_4 = 1 / C**4
        np.nan_to_num(C_inv_4, copy=False, posinf=0.0, neginf=0.0)

        lambda_val = find_optimal_lambda(T_0, X, Y, C_inv_4)

        # Шаг 4: Обновление матрицы потоков
        X_new = X + lambda_val * (Y - X)

        # Проверка критерия остановки
        relative_diff = np.abs((X_new - X) / np.where(X != 0, X, 1))
        if np.nanmax(relative_diff) < epsilon:
            break

        X = X_new
        iteration += 1

    return X.astype(int), iteration


def get_flow_matrix(G: nx.DiGraph, n: int, D: np.ndarray) -> np.ndarray:
    """
    Рассчитывает матрицу потоков на основе кратчайших путей.
    Args:
        G: Граф (транспортная сеть)
        n: Размер матрицы
        D: Матрица спроса
    Returns:
        Матрица потоков
    """
    X = np.zeros((n, n))
    no_path_count = 0

    for i in range(n):
        for j in range(n):
            if i != j and D[i, j] != 0:
                try:
                    path = nx.shortest_path(G, i, j, weight='weight')
                    for k in range(len(path)-1):
                        X[path[k], path[k+1]] += D[i, j]
                except nx.NetworkXNoPath:
                    no_path_count += 1

    return X.astype(int)


def find_optimal_lambda(T: np.ndarray, X: np.ndarray, Y: np.ndarray, C_inv_4: np.ndarray, 
                       h: float = 0.001) -> float:
    """
    Находит оптимальный шаг lambda методом бисекции.

    Args:
        T: Матрица затрат
        X: Текущая матрица потоков
        Y: Новая матрица потоков
        C_inv_4: Матрица 1/C^4
        h: Шаг для поиска

    Returns:
        Оптимальное значение lambda
    """
    delta = Y - X
    best_lambda = 0.5

    def objective(l):
        flow = X + l * delta
        return (T * delta + 0.15 * T * C_inv_4 * flow**4 * delta).sum()
    prev_val = objective(0)
    for l in np.arange(h, 1+h, h):
        current_val = objective(l)
        if np.sign(prev_val) != np.sign(current_val):
            best_lambda = l - h/2
            break
        prev_val = current_val

    return best_lambda