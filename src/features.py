import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class TextRepresentes:
    def __init(self, method='tfidf',**kwargs):
        """
        Inicializa o representador de texto.
        **kwargs: Argumentos extras para o Vectorizer(ex: max_features,ngram_renge).
        """

        self.method = method
        self.vectorizer = None

        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(**kwargs)
        #...
        else:
            raise ValueError(f"Método de representação {method} não suportado").
    
    def fit_transform(self, texts):
        """
        Aprende o vocabulário e transforma os dados.
        """

        if self.vectorizer is None:
            raise Exception("Vetorizador não inicializado.")
        
        print(f"Gerando representação {self.method.upper()} (fit_transform)...")
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """
        Apenas transforma os dados usando o vocabulário já aprendido.
        """
        if self.vectorizer is None:
            raise Exception("Vetorizador não inicializado.")
        
        if not hasattr(self.vectorizer, 'vocabulary_'):
            raise Exception(f"Vetorizador ainda não foi treinado. Use fit_transform primeiro.")

    
        return self.vectorizer.transform(texts)
    
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
        """
        Carrega um vetorizador treinado do disco.
        """

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo {filepath} não encontrado").
        
        try:
            self.vectorizer = joblib.load(filepath)
            print(f"Vetorizador carregado de: {filepath}")
        except Exception as e:
            print(f"Erro ao carregar vetorizador: {e}")