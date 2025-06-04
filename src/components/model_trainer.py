import os, sys
from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# Modelling
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            logging.info("Evaluating models")
            report = {}

            for model_name, model in models.items():
                logging.info(f"Training {model_name}")
                model_params = params.get(model_name, {})
                logging.info(f"Parameter grid for {model_name}: {model_params}")

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=model_params,
                    cv=3,
                    n_jobs=-1,
                    verbose=2
                )
                
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                report[model_name] = test_model_score

            return report
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Training the model")
            models = {
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                # "RandomForestRegressor": RandomForestRegressor(),
                # "AdaBoostRegressor": AdaBoostRegressor(),
                # "LinearRegression": LinearRegression(),
                # "Ridge": Ridge(),
                # "Lasso": Lasso(),
                # "CatBoostRegressor": CatBoostRegressor(verbose=0),
            }

            params = {
                "KNeighborsRegressor": {
                    "n_neighbors": [5,7,9,11],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree"]
                },
                "DecisionTreeRegressor": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10]
                },
                # Add parameters for other models as needed
            }

            model_report: dict = self.evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")

            best_model = models[best_model_name]
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)