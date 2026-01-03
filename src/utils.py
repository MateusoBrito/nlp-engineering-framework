import yaml
import os

def load_config(filename):
    """Lê um arquivo YAML da pasta config/"""

    config_path = os.path.join("config", filename)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuração {config_path} não encontrada.")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)