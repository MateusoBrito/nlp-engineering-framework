# src/models.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from src.utils import load_config

def get_model_config(model_name, **params):
    """
    Retorna a classe do modelo e o dicionário de parâmetros para o GridSearch.
    """
    all_configs = load_config("models.yaml")
    params = all_configs.get(model_name, {})

    if model_name == 'knn':
        return KNeighborsClassifier, params
    
    elif model_name == 'svm':
        return SVC, params
        
    elif model_name == 'dt':
        return DecisionTreeClassifier, params
    
    else:
        raise ValueError(f"Modelo '{model_name}' não configurado em src/models.py")