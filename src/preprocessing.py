import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import os

def download_nltk_resources():
    resources = ['stopwords', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4', 'punkt']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{resource}')
                except LookupError:
                    print(f"Baixando recurso NLTK: {resource}...")
                    nltk.download(resource, quiet=True)

download_nltk_resources()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def load_custom_stopwords(filepath):
    """
    Lê um arquivo .txt (uma palavra por linha) e adiciona ao set global de stopwords.
    """
    global stop_words
    
    if not filepath or not os.path.exists(filepath):
        print(f"Aviso: Arquivo de stopwords '{filepath}' não encontrado. Ignorando.")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            custom_words = {line.strip().lower() for line in f if line.strip()}
            
        initial_count = len(stop_words)
        stop_words.update(custom_words)
        final_count = len(stop_words)
        
        print(f"Stopwords personalizadas carregadas: {len(custom_words)}")
        print(f"Total de stopwords atual: {final_count} (Adicionadas: {final_count - initial_count})")
        
    except Exception as e:
        print(f"Erro ao ler arquivo de stopwords: {e}")

def remove_stopwords(tokens):
    """Remove stopwords de uma lista de tokens."""
    return [word for word in tokens if word not in stop_words]

def remove_urls(text):
    """Remove URLs do texto."""
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_tags(text):
    """Remove tags como #hashtag ou @usuario."""
    return re.sub(r'#\w+|@\w+', '', text)

def remove_punctuation(text):
    """Remove pontuação."""
    return ''.join(char for char in text if char not in string.punctuation)

def remove_numbers(text):
    """Remove dígitos numéricos."""
    return re.sub(r'\d+', '', text)

def remove_repetitions(text):
    """Remove caracteres repetidos mais de 2 vezes (ex: 'oiii' -> 'oii')."""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def remove_small(tokens, n):
    """Remove palavras com menos de N caracteres."""
    return [word for word in tokens if len(word) >= n]

def get_wordnet_pos(tag):
    """Mapeia tags POS do NLTK para tags do WordNet."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_tokens(tokens):
    """Aplica lematização baseada em POS tagging."""
    # Nota: pos_tag espera uma lista de tokens, não uma string
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]

def preprocessing(text,n=3):
    """
    Função principal que orquestra todo o pipeline de limpeza.
    Recebe uma string crua e retorna uma string limpa.
    """
    text = str(text).lower()
    text = remove_urls(text)
    text = remove_tags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_repetitions(text)
    
    tokens = text.split()
    
    tokens = remove_stopwords(tokens)
    tokens = remove_small(tokens, n)
    tokens = lemmatize_tokens(tokens)
    
    return ' '.join(tokens)