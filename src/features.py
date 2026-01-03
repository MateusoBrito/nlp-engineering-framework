import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils import load_config

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


        valid_methods = ['tfidf', 'word2vec', 'fasttext']
        if self.method not in valid_methods:
            raise ValueError(f"Método de representação '{method}' não suportado.")
    
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
    
    def save(self,filepath):
        """
        Salva o vetorizador treinado em um arquivo .pkl ou .joblib
        """
        if self.vectorizer is None:
            print("Nada a salvar.")
            return
        
        os.makedirs(os.path.dirname(filepath), exists_ok=True)

        try:
            joblib.dump(self.vectorizer, filepath)
            print(f"Vetorizador salve em: {filepath}")
        except Exception as e:
            print(f"Erro ao salvar vetorizador: {e}")
    
    def load(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo {filepath} não encontrado.")
        
        try:
            self.vectorizer = joblib.load(filepath)
            
            if hasattr(self.vectorizer, 'vector_size'):
                self.vector_size = self.vectorizer.vector_size
                
            print(f"Vetorizador carregado de: {filepath}")
        except Exception as e:
            print(f"Erro ao carregar vetorizador: {e}")