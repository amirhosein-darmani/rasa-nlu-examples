import pathlib
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.core.interpreter import RasaNLUInterpreter


def load_interpreter(model_dir, model):
    path_str = str(pathlib.Path(model_dir) / model)
    model = get_validated_path(path_str, "model")
    model_path = get_model(model)
    _, nlu_model = get_model_subdirectories(model_path)
    return RasaNLUInterpreter(nlu_model)


class RasaClassifier(BaseEstimator, ClassifierMixin):
    """
    The RasaClassifier takes a pretrained Rasa model and turns it into a scikit-learn compatible estimator.
    It expects text as input and it will predict an intent class.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        folder = str(pathlib.Path(self.model_path).parents[0])
        file = str(pathlib.Path(self.model_path).parts[-1])
        self.interpreter = load_interpreter(folder, file)
        self.class_names_ = [
            i["name"] for i in self.fetch_info_from_message("hello")["intent_ranking"]
        ]

    def fit(self, X, y):
        return self

    def fetch_info_from_message(self, text_input):
        return self.interpreter.interpreter.parse(text_input)

    def predict(self, X):
        return np.array([self.fetch_info_from_message(x)["intent"]["name"] for x in X])

    def predict_proba(self, X):
        result = []
        for x in X:
            ranking = self.fetch_info_from_message(x)["intent_ranking"]
            ranking_dict = {i["name"]: i["confidence"] for i in ranking}
            result.append([ranking_dict[n] for n in self.class_names_])
        return np.array(result)