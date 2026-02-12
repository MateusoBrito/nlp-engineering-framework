import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import load_config

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from gensim.models import Word2Vec, FastText
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

class TextRepresenter:
    def __init__(self, method='tfidf'):
        """
        Inicializa o representador de texto.
        **kwargs: Argumentos extras para o Vectorizer(ex: max_features,ngram_renge).
        """

        self.method = method.lower()

        all_configs = load_config("features.yaml")
        self.params = all_configs.get(self.method, {})

        if 'ngram_range' in self.params:
            self.params['ngram_range'] = tuple(self.params['ngram_range'])

        self.vector_size = self.params.get('vector_size', 100)
        self.vectorizer = None

        if self.method in ['word2vec', 'fasttext']:
            if not GENSIM_AVAILABLE:
                raise ImportError("Gensim não instalado. Instale com: pip install gensim")
        if 'bert' in self.method and not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers/Torch não instalado. Instale com: pip install transformers torch")

    def fit_transform(self, texts):
        """
        Aprende o vocabulário e transforma os dados.
        """

        print(f"Gerando representação {self.method.upper()} (fit_transform)...")

        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(**self.params)
            return self.vectorizer.fit_transform(texts)

        elif self.method == 'word2vec':
            tokenized_data = [text.split() for text in texts]
            self.vectorizer = Word2Vec(sentences=tokenized_data, **self.params)
            return self._get_mean_vectors(tokenized_data)
            
        elif self.method == 'fasttext':
            tokenized_data = [text.split() for text in texts]
            self.vectorizer = FastText(sentences=tokenized_data, **self.params)
            return self._get_mean_vectors(tokenized_data)
        
        elif 'static' in self.method:
            self.vectorizer = "bert_model"
            return self.transform(texts)
        
        else:
            raise ValueError(f"Método de representação '{self.method}' não suportado.")

    
    def transform(self, texts):
        """
        Apenas transforma os dados usando o vocabulário já aprendido.
        """
        if self.vectorizer is None:
            raise Exception("Vetorizador não inicializado ou não treinado. Use fit_transform primeiro.")
        
        if self.method == 'tfidf':
            if not hasattr(self.vectorizer, 'vocabulary_'):
                raise Exception("Vetorizador TF-IDF não treinado.")

            return self.vectorizer.transform(texts)
            
        elif self.method in ['word2vec', 'fasttext']:
            if not hasattr(self.vectorizer, 'wv'):
                 raise Exception("Modelo Gensim não treinado.")
            
            tokenized_data = [text.split() for text in texts]
            return self._get_mean_vectors(tokenized_data)
        
        elif 'static' in self.method:
            return self._get_transformer_embeddings(texts)
        
        else:
            raise ValueError(f"Método de representação '{self.method}' não suportado.")

    def _get_transformer_embeddings(self, texts):
        """
        Extrai embeddings usando a biblioteca Transformers.
        """
        device = 0 if torch.cuda.is_available() else -1
        device = -1
        model_name = self.params.get('model_name', 'bert-base-uncased')
        
        # Inicializa o pipeline de extração de features
        extractor = pipeline('feature-extraction', model=model_name, device=device)
        
        print(f"Extraindo embeddings com {model_name} na {'GPU' if device == 0 else 'CPU'}...")
        
        # Converte para lista se for Series/Array
        texts = list(texts)
        
        # O pipeline retorna uma lista de listas de tensores: [camada][token][dimensão]
        # Pegamos o primeiro token [0] (geralmente [CLS]) da última camada.
        raw_outputs = extractor(texts)
        
        # Extrai o vetor do token [CLS] para cada texto
        embeddings = np.array([out[0][0] for out in raw_outputs])
        
        return embeddings
    
    def _get_mean_vectors(self, tokenized_data):
        """
        Calcula a média dos vetores das palavras para representar a frase.
        """
        matrix = []
        for tokens in tokenized_data:
            vectors = []
            for word in tokens:
                # Verifica se a palavra existe no vocabulário do modelo
                if word in self.vectorizer.wv:
                    vectors.append(self.vectorizer.wv[word])
            
            if vectors:
                # Média dos vetores das palavras
                mean_vec = np.mean(vectors, axis=0)
                matrix.append(mean_vec)
            else:
                # Se nenhuma palavra for conhecida, retorna vetor de zeros
                matrix.append(np.zeros(self.vector_size))
        
        return np.array(matrix)
    
    def save_embeddings(self,filepath):
        """
        Salva o vetorizador treinado em um arquivo .pkl ou .joblib
        """
        if self.vectorizer is None:
            print("Nada a salvar.")
            return
        
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if hasattr(embeddings, "toarray"):
                embeddings = embeddings.toarray()
            
            np.save(filepath, embeddings)
            print(f"Embeddings salvos com sucesso em: {filepath}")
        except Exception as e:
            print(f"Erro ao salvar embeddings: {e}")

    def load_embeddings(self, filepath):
        """
        Carrega embeddings de um arquivo .npy.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo de embeddings não encontrado: {filepath}")
        
        try:
            embeddings = np.load(filepath)
            print(f"Embeddings carregados de: {filepath} (Shape: {embeddings.shape})")
            return embeddings
        except Exception as e:
            print(f"Erro ao carregar embeddings: {e}")
            return None