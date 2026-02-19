source venv/bin/activate

CONFIG_DIR="results/movie_review"

echo "Iniciando experimentos para dataset mpqa..."

for config_file in "$CONFIG_DIR"/config_*.yaml; do
    echo "===================================================="
    echo "EXECUTANDO: $config_file"
    echo "===================================================="
    
    python3 main.py --config "$config_file" >> logs_execucao.txt 2>&1
    
    echo "Concluído: $config_file"
    echo "----------------------------------------------------"
done

echo "Todos os experimentos foram concluídos!"