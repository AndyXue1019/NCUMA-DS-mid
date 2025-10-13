from typing import Tuple, List, Dict

import numpy as np


def adaboost_train(data_train: np.ndarray, label_train: np.ndarray, T: int) -> Tuple[List[Dict], List[float]]:
    num_samples = data_train.shape[0]
    num_features = data_train.shape[1]
    classes = np.unique(label_train)
    num_classes = len(classes)

    D = np.ones((num_samples, 1)) / num_samples  # 初始化樣本權重

    stumps = []  # 儲存每個決策樹樁的參數
    alphas = []  # 儲存每個決策樁的權重

    for m in range(T):
        best_stump = None
        best_error = float('inf')
        best_pred = None

        # 遍歷每個特徵
        for feature_i in range(num_features):
            feature_values = np.unique(data_train[:, feature_i])
            thresholds = (feature_values[:-1] + feature_values[1:]) / 2  # 計算閾值

            for threshold in thresholds:
                for inequality in ['lt', 'gt']:
                    # 嘗試為分割出的兩個區域分配最優的類別
                    for c1 in classes:
                        for c2 in classes:
                            if c1 == c2: 
                                continue

                            preds = np.empty(num_samples, dtype=object)
                            if inequality == 'lt':
                                preds[data_train[:, feature_i] <= threshold] = c1
                                preds[data_train[:, feature_i] > threshold] = c2
                            else: # gt
                                preds[data_train[:, feature_i] > threshold] = c1
                                preds[data_train[:, feature_i] <= threshold] = c2

                            errors = (preds != label_train).astype(float)
                            weighted_error = np.dot(D.T, errors)

                            if weighted_error < best_error:
                                best_error = weighted_error
                                best_stump = {
                                    'feature_index': feature_i,
                                    'threshold': threshold,
                                    'inequality': inequality,
                                    'class1': c1,
                                    'class2': c2
                                }
                                best_pred = preds

        # 計算弱分類器的權重 (AdaBoost)
        epsilon = best_error

        if epsilon >= 1 - (1 / num_classes):
            # 如果錯誤率太高，提前停止訓練
            print(f'訓練在第 {m + 1} 輪停止，錯誤率過高: {epsilon:.2f}')
            break

        beta = epsilon / (1 - epsilon)
        alpha = np.log(1 / beta)

        # 更新樣本權重
        matches = (best_pred == label_train).astype(float)
        D *= np.power(beta, matches).reshape(-1, 1)
        D /= np.sum(D)  # 歸一化

        stumps.append(best_stump)
        alphas.append(alpha)

        if best_error == 0:
            print(f'訓練在第 {m + 1} 輪達到完美分類。')
            break

    return stumps, alphas


def adaboost_predict(X: np.ndarray, stumps: list, alphas: list) -> np.ndarray:
    num_samples = X.shape[0]
    classes = np.unique([s['class1'] for s in stumps] + [s['class2'] for s in stumps])
    class_votes = {c: np.zeros(num_samples) for c in classes}

    for alpha, stump in zip(alphas, stumps):
        feature_index = stump['feature_index']
        threshold = stump['threshold']
        inequality = stump['inequality']
        c1 = stump['class1']
        c2 = stump['class2']

        preds = np.empty(num_samples, dtype=object)
        if inequality == 'lt':
            preds[X[:, feature_index] <= threshold] = c1
            preds[X[:, feature_index] > threshold] = c2
        else: # gt
            preds[X[:, feature_index] > threshold] = c1
            preds[X[:, feature_index] <= threshold] = c2
        
        for c in classes:
            class_votes[c] += alpha * (preds == c).astype(float)

    # 找出每個樣本得票最高的類別
    final_preds = np.array([max(class_votes, key=lambda c: class_votes[c][i]) for i in range(num_samples)])
    
    return final_preds