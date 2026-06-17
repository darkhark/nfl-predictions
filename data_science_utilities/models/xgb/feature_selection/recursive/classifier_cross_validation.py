from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss, f1_score, precision_score,
    recall_score, accuracy_score
)
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


class ClassifierCrossValidationRecursiveFeatureSelection:
    HIGHER_IS_BETTER_METRICS = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']

    _RANDOM_SEED = 32

    def __init__(self, X_train, y_train, xgb_params, label_encoder=None, model_score_metric='roc_auc'):
        self.X_train = X_train
        self.y_train = y_train
        self.xgb_params = xgb_params
        self.label_encoder = label_encoder
        self.model_score_metric = model_score_metric
        self.features = self.X_train.columns
        self.all_features = {}
        self.test_preds = {}
        self.all_models = {}
        self.all_model_scores = []
        self.importances = None

    def get_optimal_features_no_grouped_records(self, drop_rate=.1, max_iter=10, verbose=0, base_margin=None,
                                                n_folds=5, min_features=None, on_iteration=None):
        """
        Run cross-validated RFE for up to max_iter iterations, dropping drop_rate of the
        surviving features each round. When min_features is set, the drop is clamped so
        the feature count never falls below it and the loop stops once it is reached —
        the same floor semantics as BartBackwardElimination.

        on_iteration, when provided, is called after each completed iteration with a
        dict (iteration, max_iter, num_features, score). Headless nbconvert buffers a
        cell's stdout until the cell finishes, so callers wanting LIVE progress should
        have the callback write to a sidecar file rather than rely on verbose prints.
        """
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
            for fold, (train_index, test_index) in enumerate(strat_k_fold.split(self.X_train, self.y_train)):
                X_train, X_test = self.X_train.iloc[train_index], self.X_train.iloc[test_index]
                y_train, y_test = self.y_train.iloc[train_index], self.y_train.iloc[test_index]
                model.fit(X_train[train_features], y_train, eval_set=[(X_test[train_features], y_test)], verbose=False)
                test_preds = model.predict_proba(X_test[train_features])[:, 1]
                self._get_non_zero_importances(model)
                model_scores.append(self._get_test_scores(test_preds, y_test))
                if len(train_features) in self.all_models:
                    if fold in self.all_models[len(train_features)]:
                        self.all_models[len(train_features)][fold].append(model)
                        self.test_preds[len(train_features)][fold].append(test_preds)
                    else:
                        self.all_models[len(train_features)][fold] = model
                        self.test_preds[len(train_features)][fold] = test_preds
                else:
                    self.all_models[len(train_features)] = {fold: model}
                    self.test_preds[len(train_features)] = {fold: test_preds}
            self.all_model_scores.append(np.mean(model_scores))
            if on_iteration is not None:
                on_iteration({
                    'iteration': i + 1,
                    'max_iter': max_iter,
                    'num_features': len(train_features),
                    'score': self.all_model_scores[-1],
                })
            if min_features is not None and len(train_features) <= min_features:
                break
            if i == 0:
                train_features = list(self.importances.index)
            else:
                new_num_features = int(np.floor(len(train_features) * (1 - drop_rate)))
                if min_features is not None:
                    new_num_features = max(new_num_features, min_features)
                train_features = list(self.importances.index)[:new_num_features]

    def get_features_in_dataframe(self):
        """
        Get the features in a DataFrame where the index is the number of features and the columns are the feature names

        :return: A DataFrame of the features
        """
        return pd.DataFrame.from_dict(self.all_features, orient='index')

    def get_best_num_features(self, min_diff):
        """
        This method essentially tries to find the smallest number of features that can achieve a model score within
        min_diff of the best possible score. This helps to avoid overfitting by using too many features.

        :param min_diff: The minimum difference between the best score and the next best score
        :return: The number of features that resulted in the best score with the smallest number of features
        """
        best_score = 0
        best_num_feats = 0
        feats_and_scores = self._get_metric_values()
        feats_and_scores.sort_index(inplace=True)
        past_scores = {}
        for curr_num_feats, scores in feats_and_scores.iterrows():
            curr_score = scores[0]
            past_scores[curr_num_feats] = curr_score
            if self._is_curr_score_better(curr_score, best_score, min_diff):
                best_num_feats, best_score = self._update_best_score(
                    past_scores, curr_score, curr_num_feats, min_diff
                )
        return best_num_feats

    def _is_curr_score_better(self, curr_score, best_score, min_diff):
        if self.model_score_metric in self.HIGHER_IS_BETTER_METRICS:
            is_better_score = curr_score > best_score
            is_gt_min_diff = (curr_score - best_score) > min_diff
        else:
            is_better_score = curr_score < best_score
            is_gt_min_diff = (best_score - curr_score) > min_diff
        return is_better_score and is_gt_min_diff

    def _update_best_score(self, past_scores, curr_score, curr_num_feats, min_diff):
        """
        Update the best score and number of features if the current score is better than the best score
        and the difference between the current score and the best score is greater than min_diff.

        If a past score is within min_diff of the current score, then the best score and number of features
        are updated to the past score and number of features.

        :param past_scores: A dictionary of scores already iterated over where the key is the number
        of features and the value is the score
        :param curr_score: The current score in the iteration
        :param curr_num_feats: The current number of features in the iteration
        :param min_diff: The threshold for the difference between the best score and the current score
        :return: The best number of features and the best score given the threshold and reduced model complexity
        """
        best_num_feats = curr_score
        best_score = curr_num_feats
        for past_num_feats, past_score in past_scores.items():
            if self.model_score_metric in self.HIGHER_IS_BETTER_METRICS:
                is_past_score_diff_within_min_diff = curr_score - past_score < min_diff
            else:
                is_past_score_diff_within_min_diff = past_score - curr_score < min_diff
            if is_past_score_diff_within_min_diff:
                best_num_feats = past_num_feats
                best_score = past_score
                break
        return best_num_feats, best_score

    def plot_model_scores(self, **fig_kw):
        """
        Plots the model scores for each iteration of RFE

        :param fig_kw: Keyword arguments passed to plt.figure
        :return: None
        """
        if 'marker' not in fig_kw:
            fig_kw['marker'] = 'o'
        ax = self._get_metric_values().plot(**fig_kw)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel(self.model_score_metric)
        return plt.gcf(), ax

    def _get_metric_values(self):
        # Get the class names, for example, '0' and '1' for binary classification to use as column names
        if len(self.all_models):
            first_key = list(self.all_models.keys())[0]
            if self.label_encoder is None:
                class_names = self.all_models[first_key][0].classes_
            else:
                class_names = self.label_encoder.inverse_transform(self.all_models[first_key][0].classes_)
            df = pd.DataFrame(self.all_model_scores, index=list(self.all_features.keys()))
            return df
        else:
            return None

    def _get_test_scores(self, preds, y_test):
        if self.model_score_metric == 'roc_auc':
            score = roc_auc_score(y_test, preds)
        elif self.model_score_metric == 'log_loss':
            score = log_loss(y_test, preds)
        elif self.model_score_metric == 'brier':
            score = brier_score_loss(y_test, preds)
        elif self.model_score_metric == 'f1':
            score = f1_score(y_test, preds.round())
        elif self.model_score_metric == 'precision':
            score = precision_score(y_test, preds.round())
        elif self.model_score_metric == 'recall':
            score = recall_score(y_test, preds.round())
        elif self.model_score_metric == 'accuracy':
            score = accuracy_score(y_test, preds.round())
        else:
            raise ValueError('model_score_metric must be one of roc_auc, log_loss, brier, f1, precision, recall, accuracy')
        return score

    def _get_non_zero_importances(self, model):
        importances = pd.Series(model.get_booster().get_score(), name='Feature Importance')
        importances = importances.T.sort_values(ascending=False)
        self.importances = importances[importances > 0]
