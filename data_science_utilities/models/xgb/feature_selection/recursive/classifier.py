from sklearn.metrics import (
    roc_auc_score, log_loss, f1_score, precision_score,
    recall_score, accuracy_score
)
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ClassifierRecursiveFeatureElimination:
    HIGHER_IS_BETTER_METRICS = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']

    def __init__(self, X_train, y_train, X_test, y_test, xgb_params, label_encoder=None, model_score_metric='roc_auc'):
        """
        Recursive Feature Elimination (RFE) is a feature selection algorithm that selects features by recursively
        considering smaller and smaller sets of features. It starts with all features and removes the least important
        feature until the specified number of features is reached. The importance of the features is determined by
        overfitting a model and evaluating the feature importance.

        :param X_train: A dataframe of the training data
        :param y_train: A dataframe of the training target
        :param X_test: A dataframe of the testing data
        :param y_test: A dataframe of the testing target
        :param xgb_params: A dictionary of XGBoost parameters where the model is slightly overfits
        :param label_encoder: sklearn.preprocessing.LabelEncoder object to encode the target column if needed
        :param model_score_metric: The metric to use to evaluate the model and plot. Default is 'roc_auc'
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.xgb_params = xgb_params
        self.label_encoder = label_encoder
        self.model_score_metric = model_score_metric
        self.features = self.X_train.columns
        self.all_features = {}
        self.test_preds = []
        self.all_models = []
        self.all_model_scores = []
        self.importances = None

    def get_optimal_features(self, drop_rate=.1, max_iter=10, verbose=0, base_margin=None):
        """
        Removes features based on a specified decay rate until the maximum number of iterations is reached. It then
        recalculates the feature importance and evaluates the model on the specified metric.

        :param drop_rate: The proportion of remaining features to drop with each round of RFE
        :param max_iter: The maximum number of iterations to run RFE
        :param verbose: The level of verbosity of the model training
        :param base_margin: The warm start predictions for the model if desired
        :return:
        """
        max_iter = min(max_iter, len(self.features))
        train_features = self.features
        for i in range(max_iter):
            if verbose:
                print(f"Running iteration {i + 1} of RFE with {len(train_features)} features")
            self.all_features[len(train_features)] = train_features
            if 'eval_metric' not in self.xgb_params:
                raise ValueError('The xgb_params must have an eval_metric key')
            model = XGBClassifier(**self.xgb_params)
            eval_set = [(self.X_train[train_features], self.y_train), (self.X_test[train_features], self.y_test)]
            model.fit(
                self.X_train[train_features],
                self.y_train,
                eval_set=eval_set,
                verbose=verbose,
                base_margin=base_margin
            )
            self.all_models.append(model)
            test_preds = model.predict_proba(self.X_test[train_features])
            self.test_preds.append(test_preds)
            self._get_non_zero_importances(model)
            self.all_model_scores.append(self._get_test_metric_values(test_preds))
            if i == 0:
                train_features = list(self.importances.index)
            else:
                new_num_features = int(np.floor(len(train_features) * (1 - drop_rate)))
                train_features = list(self.importances.index)[:new_num_features]

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

    def _get_metric_values(self):
        # Get the class names, for example, '0' and '1' for binary classification to use as column names
        if len(self.all_models):
            if self.label_encoder is None:
                class_names = self.all_models[0].classes_
            else:
                class_names = self.label_encoder.inverse_transform(self.all_models[0].classes_)
            df = pd.DataFrame(self.all_model_scores, index=list(self.all_features.keys()), columns=class_names)
            # remove all columns after the first if the target column is binary
            if len(class_names) == 2:
                df.drop(columns=class_names[1], inplace=True)
            return df
        else:
            return None

    def _get_non_zero_importances(self, model):
        """
        Calculate 'self.importances' attribute from booster

        :param model: Trained XGBoost model
        :return: pd.Series of feature importances
        """
        importances = pd.Series(model.get_booster().get_score(), name='Feature Importance')
        importances = importances.T.sort_values(ascending=False)
        self.importances = importances[importances > 0]

    def _get_test_metric_values(self, test_preds):
        if self.model_score_metric == 'roc_auc':
            return [roc_auc_score(self.y_test == i, test_preds[:, i]) for i in range(test_preds.shape[1])]
        elif self.model_score_metric == 'log_loss':
            return [log_loss(self.y_test == i, test_preds[:, i]) for i in range(test_preds.shape[1])]
        elif self.model_score_metric == 'accuracy':
            return [accuracy_score(self.y_test, np.argmax(test_preds, axis=1))]
        elif self.model_score_metric == 'f1':
            return [f1_score(self.y_test, np.argmax(test_preds, axis=1), average='weighted')]
        elif self.model_score_metric == 'precision':
            return [precision_score(self.y_test, np.argmax(test_preds, axis=1), average='weighted')]
        elif self.model_score_metric == 'recall':
            return [recall_score(self.y_test, np.argmax(test_preds, axis=1), average='weighted')]
        elif self.model_score_metric == 'error':
            return [1 - accuracy_score(self.y_test, np.argmax(test_preds, axis=1))]
        else:
            raise ValueError(f"Unsupported metric {self.model_score_metric}")
