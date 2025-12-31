import argparse
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import TextPreprocessor
from src.features import TextRepresenter
from src.validation import get_validation_strategy
from src.models import get_model_config
from src.trainer import ModelTrainer
from src.evaluation import evaluate_model, save_results, plot_confusion_matrix
 
parser = argparse.ArgumentParser(description="Pipeline de Classificação de Texto")
parser.add_argument('--dataset', type=str, required=True, help="Caminho para o dataset (CSV)")
parser.add_argument('--text_col', type=str, default='message', help="Nome da coluna de texto")
parser.add_argument('--label_col', type=str, default='label', help="Nome da coluna de rótulo")
parser.add_argument('--model', type=str, required=True, choices=['knn', 'svm', 'dt'], help="Modelo a utilizar (knn,csvm,dt)")
parser.add_argument('--representation', type=str, default='tfidf', help="Método de representação (tfidf, word2vec, etc)")
parser.add_argument('--validation', type=str, default='stratified_kfold', choices=['kfold', 'stratified_kfold', 'holdout'], help="Estratégia de validação")
parser.add_argument('--n_splits', type=int, default=5, help="Número de folds")
parser.add_argument('--output_dir', type=str, default='results', help="Diretório para salvar resultados")

args = parser.parse_args()

def main():
    print(f"\n=== Iniciando pipeline de classificar ===")
    print(f"Dataset: {args.dataset}")
    print(f"Modelo: {args.model.upper()}")
    print(f"Representação: {args.representation.upper()}")

    try:
        try:
            df = pd.read_csv(args.dataset)
        except:
            df = pd.read_csv(args.dataset, sep=';')
            
        print(f"Dados carregados: {len(df)} linhas.")
    except Exception as e:
        print(f"Erro ao carregar dataset: {e}")
        return
    
    print("Executando pré-processamento...")
    preprocessor = TextPreprocessor()

    clean_text_col = 'clean_text'
    df[clean_text_col] = df[args.text_col].apply(preprocessor.preprocess)

    X = df[clean_text_col].values
    y = df[args.label_col].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = [str(cls) for cls in le.classes_]

    splitter = get_validation_strategy(args.validation, n_splits=args.n_splits)

    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y_encoded)):
        print(f"\n--- Fold {fold + 1} ---")
        
        # Divisão dos dados (Texto cru)
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        # Vetorização (Feature Extraction)
        # Importante: Fit apenas no treino para evitar vazamento de dados
        representer = TextRepresenter(method=args.representation)
        
        try:
            X_train_vec = representer.fit_transform(X_train_raw)
            X_test_vec = representer.transform(X_test_raw)
        except Exception as e:
            print(f"Erro na vetorização: {e}")
            continue
        
        # Configuração do Modelo
        model_class, param_grid = get_model_config(args.model)
        
        # Instancia o Treinador
        trainer = ModelTrainer(model_class, param_grid, cv=3) # CV interno para GridSearch
        
        # Otimização de Hiperparâmetros
        print("Otimizando parâmetros...")
        trainer.optimize_hyperparameters(X_train_vec, y_train)
        
        # Treinamento Final (com melhores parâmetros)
        trainer.train(X_train_vec, y_train)
        
        # Predição
        y_pred = trainer.predict(X_test_vec)

        metrics = evaluate_model(y_test, y_pred, labels=class_names)
        print(f"Acurácia: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f}")
        
        # Salvar Resultados deste fold
        fold_info = {
            'dataset': os.path.basename(args.dataset),
            'model': args.model,
            'representation': args.representation,
            'fold': fold + 1,
            'best_params': str(trainer.best_params)
        }
        
        # Caminho do arquivo de log geral
        log_path = os.path.join(args.output_dir, 'experiment_log.csv')
        os.makedirs(args.output_dir, exist_ok=True)
        
        save_results(metrics, fold_info, log_path)
        
        # Salvar Matriz de Confusão (Opcional)
        cm_path = os.path.join(args.output_dir, 'plots', f'cm_{args.model}_{fold+1}.png')
        plot_confusion_matrix(y_test, y_pred, labels=class_names, filepath=cm_path)
        
        fold_results.append(metrics['accuracy'])

    # Resumo final
    if fold_results:
        print(f"\n=== Média Final: {np.mean(fold_results):.4f} (+/- {np.std(fold_results):.4f}) ===")

if __name__ == "__main__":
    main()  