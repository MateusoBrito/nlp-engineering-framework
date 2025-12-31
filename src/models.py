# src/models.py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def get_model_config(model_name, **params):
    """
    Retorna a classe do modelo e o dicionário de parâmetros para o GridSearch.
    """
    if model_name == 'knn':
        return KNeighborsClassifier, {
            'n_neighbors': [3, 5, 7, 11],
            'metric': ['euclidean', 'manhattan']
        }
    
    elif model_name == 'svm':
        return SVC, {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
        
    elif model_name == 'dt':
        return DecisionTreeClassifier, {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    
    else:
        raise ValueError(f"Modelo '{model_name}' não configurado em src/models.py")