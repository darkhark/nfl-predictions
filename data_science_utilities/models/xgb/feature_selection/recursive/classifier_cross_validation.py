from sklearn.metrics import (
    roc_auc_score, log_loss, f1_score, precision_score,
    recall_score, accuracy_score
)
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


class ClassifierCrossValidationRecursiveFeatureSelection:
    _RANDOM_SEED = 32

    def __init__(self, X_train, y_train, xgb_params, label_encoder=None, model_score_metric='roc_auc'):
        self.X_train = X_train
        self.y_train = y_train
        self.xgb_params = xgb_params
        self.label_encoder = label_encoder
        self.model_score_metric = model_score_metric
        self.features = self.X_train.columns
        self.all_features = {}
        self.test_preds = []
        self.all_models = {}
        self.all_model_scores = []
        self.importances = None

    def get_optimal_features_no_grouped_records(self, drop_rate=.1, max_iter=10, verbose=0, base_margin=None, n_folds=5):
        max_iter = min(max_iter, len(self.features))
        train_features = self.features
        for i in range(max_iter):
            if verbose > 0:
                print(f'Iteration {i+1} of {max_iter}')
            self.all_features[len(train_features)] = train_features
            if 'eval_metric' not in self.xgb_params:
                raise ValueError('The xgb_params must have an eval_metric key')
            # begin cross validation, where the mean score is calculated to determine the best features
            model = XGBClassifier(**self.xgb_params)
            # create cross validation folds
            strat_k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self._RANDOM_SEED)
            model_scores = []
            for train_index, test_index in strat_k_fold.split(self.X_train, self.y_train):
                X_train, X_test = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
                y_train, y_test = self.y_train.iloc[train_index], self.y_train.iloc[test_index]
                model.fit(X_train[train_features], y_train, eval_set=[(X_test[train_features], y_test)], verbose=False)
                preds = model.predict_proba(X_test[train_features])[:, 1]
                model_scores.append(self._get_test_scores(preds, y_test))
                if len(train_features) in self.all_models:
                    self.all_models[len(train_features)].append(model)
                else:
                    self.all_models[len(train_features)] = [model]
            self.all_model_scores.append(np.mean(model_scores))

    def _get_test_scores(self, preds, y_test):
        if self.model_score_metric == 'roc_auc':
            score = roc_auc_score(y_test, preds)
        elif self.model_score_metric == 'log_loss':
            score = log_loss(y_test, preds)
        elif self.model_score_metric == 'f1':
            score = f1_score(y_test, preds.round())
        elif self.model_score_metric == 'precision':
            score = precision_score(y_test, preds.round())
        elif self.model_score_metric == 'recall':
            score = recall_score(y_test, preds.round())
        elif self.model_score_metric == 'accuracy':
            score = accuracy_score(y_test, preds.round())
        else:
            raise ValueError('model_score_metric must be one of roc_auc, log_loss, f1, precision, recall, accuracy')
        return score