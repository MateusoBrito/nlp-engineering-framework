import os
from src.comparison import ResultsComparator

BASE_DIR = "results/movie_review"
COMP_DIR = "results/movie_review/results_comparison" 
DATASET_NAME = "Movie_Review.csv" 

REPRESENTATIONS = ['tfidf', 'word2vec', 'fasttext', 'bert_static']
MODELS = ['knn', 'svm', 'dt']

def main():
    try:
        comparator = ResultsComparator(BASE_DIR)
    except FileNotFoundError:
        print(f"ERRO: Pasta '{BASE_DIR}' não encontrada.")
        return

    print(f"=== Iniciando Análise para Dataset: {DATASET_NAME} ===\n")
    
    # =========================================================================
    # PARTE 1: Comparar MODELOS (fixando a Representação)
    # =========================================================================
    for rep in REPRESENTATIONS:
        print(f"--- Analisando Modelos em: {rep.upper()} ---")
        
        aggregated_results = {}
        for model in MODELS:
            data = comparator.get_model_data(DATASET_NAME, rep, model)
            if data: aggregated_results[model] = data

        if len(aggregated_results) < 2: continue

        # 1. Gráfico de Barras Combinado
        comparator.plot_combined_bars(
            aggregated_results, 
            title_suffix=f"{rep.upper()}",
            output_path=f"{COMP_DIR}/models/bar/bar_{rep}.png"
        )

        # 2. Gráfico de Diferenças 
        comparator.plot_pairwise_differences(
            aggregated_results,
            metric='accuracy',
            title_suffix=f"{rep.upper()}",
            output_path=f"{COMP_DIR}/models/difference/diff_{rep}.png"
        )

        # 3. Relatório de Texto
        comparator.save_statistical_report(
            aggregated_results,
            title_suffix=f"Modelos em {rep.upper()}",
            output_path=f"{COMP_DIR}/models/statistical_tests/report_{rep}.txt"
        )

    # =========================================================================
    # PARTE 2: Comparar REPRESENTAÇÕES (fixando o Modelo)
    # =========================================================================
    print("\n--- Comparando Representações ---")

    for model in MODELS:
        print(f"--- Analisando Repr. para: {model.upper()} ---")
        
        aggregated_results = {}
        for rep in REPRESENTATIONS:
            data = comparator.get_model_data(DATASET_NAME, rep, model)
            if data: aggregated_results[rep] = data

        if len(aggregated_results) < 2: continue

        # 1. Gráfico de Barras Combinado
        comparator.plot_combined_bars(
            aggregated_results, 
            title_suffix=f"{model.upper()}",
            output_path=f"{COMP_DIR}/representation/bars/bar_{model}.png"
        )
        
        # 2. Gráfico de Diferenças 
        comparator.plot_pairwise_differences(
            aggregated_results,
            metric='accuracy',
            title_suffix=f"{model.upper()}",
            output_path=f"{COMP_DIR}/representation/diff_plot/diff_{model}.png"
        )

        # 3. Relatório de Texto
        comparator.save_statistical_report(
            aggregated_results,
            title_suffix=f"Representações para {model.upper()}",
            output_path=f"{COMP_DIR}/representation/statistical_tests/report_{model}.txt"
        )

    print(f"\n=== Análise concluída! Verifique a pasta '{COMP_DIR}' ===")

if __name__ == "__main__":
    main()