from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


class OptimalXGBHyperparameterSearch:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def search(self, params, **search_kwargs):
        """
        Perform a random search for the optimal hyperparameters for the XGBoost model

        :param params: The XGBoost hyperparameters to search over
        :param search_kwargs: Additional arguments to pass to the RandomizedSearchCV object
        :return: A RandomizedSearchCV object
        """
        xgb = XGBClassifier(**params)

        random_search = RandomizedSearchCV(
            xgb,
            param_distributions=params,
            **search_kwargs
        )

        random_search.fit(
            self.X_train,
            self.y_train,
            eval_set=[
                (self.X_train, self.y_train),
                (self.X_test, self.y_test)
            ],
            verbose=1
        )

        print("\n\nBest parameters found: ", random_search.best_params_)
        print("\n\nBest score found: ", random_search.best_score_)
        print("\n\nBest Selected Features: ", random_search.best_estimator_.get_booster().feature_names)

        return random_search
