import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from src.transformer_trainer import TransformerTrainer
from src.datasets import TextDataset
from src.evaluation import evaluate_model, save_results, plot_confusion_matrix

def run_transformer_pipeline(X_raw, y_encoded, splitter, model_name, params, dataset_path, output_dir, class_names):
    fold_results = []
    
    # O Tokenizer é fixo para todos os folds
    tokenizer = BertTokenizer.from_pretrained(model_name)
    num_labels = len(class_names)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X_raw, y_encoded)):
        print(f"\n--- Fold {fold + 1} ---")
        
        # 1. Divisão dos textos brutos
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # 2. Tokenização específica para o fold
        train_encodings = tokenizer(list(X_train_raw), truncation=True, padding=True, max_length=128)
        test_encodings = tokenizer(list(X_test_raw), truncation=True, padding=True, max_length=128)

        # 3. Preparação dos DataLoaders
        train_set = TextDataset(train_encodings, y_train)
        test_set = TextDataset(test_encodings, y_test)
        
        train_loader = DataLoader(train_set, batch_size=params.get('batch_size', 16), shuffle=True)
        test_loader = DataLoader(test_set, batch_size=params.get('batch_size', 16))

        # 4. Inicialização do Treinador (Resetando o modelo para cada fold)
        trainer = TransformerTrainer(model_name, num_labels, params)

        # 5. Loop de Épocas
        for epoch in range(trainer.epochs):
            avg_loss = trainer.train(train_loader)
            print(f"Epoch {epoch+1}/{trainer.epochs} | Loss: {avg_loss:.4f}")

        y_pred = trainer.predict(test_loader)
        
        metrics = evaluate_model(y_test, y_pred, labels=class_names)
        print(f"Acurácia: {metrics['accuracy']:.4f} | F1-Macro: {metrics['f1_macro']:.4f}")

        fold_info = {
            'dataset': os.path.basename(dataset_path),
            'model': model_name,
            'representation': 'fine-tuning',
            'fold': fold + 1,
            'params': str(params)
        }
        
        log_path = os.path.join(output_dir, 'experiment_log.csv')
        save_results(metrics, fold_info, log_path)
        
        plots_path = os.path.join(output_dir, 'plots')
        cm_path = os.path.join(plots_path, f'cm_{model_name.replace("/", "_")}_{fold+1}.png')
        plot_confusion_matrix(y_test, y_pred, labels=class_names, filepath=cm_path)
        
        fold_results.append(metrics['accuracy'])

        torch.cuda.empty_cache()

    print(f"\n=== Média Final Transformer: {np.mean(fold_results):.4f} ===")
    return fold_results