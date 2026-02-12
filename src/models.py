# src/models.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.utils import load_config

def get_model_config(model_name):
    """
    Identifica e retorna a configuração do modelo.
    """
    all_configs = load_config("models.yaml")
    
    classic_configs = all_configs.get('classic_models', {})
    transformer_configs = all_configs.get('transformer_models', {})

    if model_name in classic_configs:
        params = classic_configs[model_name]
        mapping = {
            'knn': KNeighborsClassifier,
            'svm': SVC,
            'dt': DecisionTreeClassifier
        }
        return "classic", mapping[model_name], params

    elif model_name in transformer_configs:
        params = transformer_configs[model_name]
        return "transformer", model_name, params

    raise ValueError(f"Modelo '{model_name}' não encontrado em nenhuma categoria do models.yaml")