import argparse
import pandas as pd
import os
import numpy as np
import yaml  
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import TextPreprocessor
from src.validation import get_validation_strategy
from src.models import get_model_config
from src.classic_pipeline import run_classic_pipeline
from src.transformer_pipeline import run_transformer_pipeline

def main():
    parser = argparse.ArgumentParser(description="Pipeline Modular: Classic vs Transformer")
    parser.add_argument('--config', type=str, required=True, help="Caminho para o arquivo YAML")
    args_cli = parser.parse_args()

    try:
        with open(args_cli.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}")
        return

    # Extração de variáveis do YAML
    prep_cfg = config['preprocessing']
    train_cfg = config['training']
    
    dataset_path = prep_cfg['input_path']
    output_dir = train_cfg['output_dir']
    model_name = train_cfg['model']

    print(f"\n=== Iniciando orquestração do pipeline ===")
    print(f"Dataset: {dataset_path} | Modelo: {model_name.upper()}")

    try:
        df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_csv(dataset_path, sep=';')
    except Exception as e:
        print(f"Erro ao carregar dataset: {e}")
        return
    
    print("Executando Pré-processamento...")
    preprocessor = TextPreprocessor()
    clean_text_col = prep_cfg.get('output_col', 'clean_text')
    df[clean_text_col] = df[prep_cfg['text_col']].apply(preprocessor.preprocess)

    # 3. Preparação de Rótulos (Label Encoding)
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[train_cfg['label_col']])
    class_names = [str(cls) for cls in le.classes_]

    # 4. Roteamento de Modelos
    model_type, model_obj, params = get_model_config(model_name)
    splitter = get_validation_strategy(train_cfg['validation'], n_splits=train_cfg['n_splits'])

    if model_type == "classic":
        run_classic_pipeline(
            X=df[clean_text_col].values,
            y_encoded=y_encoded,
            splitter=splitter,
            representation=config['embeddings']['representation'],
            model_class=model_obj,
            param_grid=params,
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir,
            class_names=class_names
        )

    elif model_type == "transformer":
        run_transformer_pipeline(
            X_raw=df[clean_text_col].values,
            y_encoded=y_encoded,
            splitter=splitter,
            model_name=model_name, 
            params=params,
            dataset_path=dataset_path,
            output_dir=output_dir,
            class_names=class_names
        )

if __name__ == "__main__":
    main()