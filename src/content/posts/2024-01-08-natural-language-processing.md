---
title: "Natural Language Processing: From Traditional Methods to Transformers"
date: "2024-01-08"
author: "Dr. Lisa Chen"
excerpt: "Comprehensive guide to NLP techniques, from basic text processing to advanced transformer models."
tags: ["nlp", "transformers", "bert", "text-processing", "python"]
category: "Natural Language Processing"
---

# Natural Language Processing: From Traditional Methods to Transformers

Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. This field has evolved dramatically from rule-based systems to sophisticated neural networks that can engage in human-like conversations.

## Introduction to NLP

NLP combines computational linguistics with machine learning to bridge the gap between human communication and computer understanding.

### Core NLP Tasks:
- **Text Classification**: Categorizing text into predefined classes
- **Named Entity Recognition**: Identifying entities like names, locations, dates
- **Sentiment Analysis**: Determining emotional tone of text
- **Machine Translation**: Converting text between languages
- **Question Answering**: Providing answers to natural language questions
- **Text Summarization**: Creating concise summaries of longer texts

## Text Preprocessing

### Basic Text Cleaning
```python
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_filter(self, text):
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens
    
    def stem_tokens(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]
    
    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text, use_stemming=False, use_lemmatization=True):
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize and filter
        tokens = self.tokenize_and_filter(cleaned_text)
        
        # Apply stemming or lemmatization
        if use_stemming:
            tokens = self.stem_tokens(tokens)
        elif use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        
        return tokens

# Usage example
preprocessor = TextPreprocessor()
text = "I'm loving this amazing product! Check out https://example.com #awesome @company"
processed_tokens = preprocessor.preprocess(text)
print(processed_tokens)  # ['love', 'amazing', 'product', 'check']
```

### Advanced Text Normalization
```python
import spacy
from textblob import TextBlob

class AdvancedPreprocessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
    
    def extract_entities(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def pos_tagging(self, text):
        doc = self.nlp(text)
        pos_tags = [(token.text, token.pos_) for token in doc]
        return pos_tags
    
    def dependency_parsing(self, text):
        doc = self.nlp(text)
        dependencies = [(token.text, token.dep_, token.head.text) 
                       for token in doc]
        return dependencies
    
    def spell_correction(self, text):
        blob = TextBlob(text)
        return str(blob.correct())

# Example usage
advanced_preprocessor = AdvancedPreprocessor()
text = "Apple Inc. is planning to open a new store in New York next month."

entities = advanced_preprocessor.extract_entities(text)
print("Entities:", entities)
# [('Apple Inc.', 'ORG'), ('New York', 'GPE'), ('next month', 'DATE')]

pos_tags = advanced_preprocessor.pos_tagging(text)
print("POS Tags:", pos_tags[:5])
# [('Apple', 'PROPN'), ('Inc.', 'PROPN'), ('is', 'AUX'), ('planning', 'VERB'), ('to', 'PART')]
```

## Feature Engineering for Text

### Bag of Words and TF-IDF
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Sample data
texts = [
    "I love this movie, it's fantastic!",
    "This film is terrible, I hate it.",
    "Great acting and amazing storyline.",
    "Boring movie, waste of time.",
    "Excellent cinematography and direction."
]
labels = [1, 0, 1, 0, 1]  # 1: positive, 0: negative

# Bag of Words
bow_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
bow_features = bow_vectorizer.fit_transform(texts)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_features = tfidf_vectorizer.fit_transform(texts)

# Train a simple classifier
X_train, X_test, y_train, y_test = train_test_split(
    tfidf_features, labels, test_size=0.2, random_state=42
)

classifier = MultinomialNB()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
```

### N-grams and Advanced Features
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np

class AdvancedFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
            max_features=5000,
            stop_words='english'
        )
    
    def extract_statistical_features(self, texts):
        features = []
        for text in texts:
            # Basic statistics
            word_count = len(text.split())
            char_count = len(text)
            avg_word_length = np.mean([len(word) for word in text.split()])
            
            # Punctuation count
            punct_count = sum([1 for char in text if char in string.punctuation])
            
            # Capital letters count
            capital_count = sum([1 for char in text if char.isupper()])
            
            features.append([
                word_count, char_count, avg_word_length, 
                punct_count, capital_count
            ])
        
        return np.array(features)
    
    def extract_tfidf_features(self, texts):
        return self.tfidf_vectorizer.fit_transform(texts)
    
    def combine_features(self, texts):
        # Extract different types of features
        statistical_features = self.extract_statistical_features(texts)
        tfidf_features = self.extract_tfidf_features(texts).toarray()
        
        # Combine features
        combined_features = np.hstack([statistical_features, tfidf_features])
        return combined_features
```

## Word Embeddings

### Word2Vec Implementation
```python
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import numpy as np

class Word2VecEmbeddings:
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
    
    def train(self, sentences):
        # sentences should be a list of tokenized sentences
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1  # Skip-gram model
        )
        return self.model
    
    def get_word_vector(self, word):
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        return np.zeros(self.vector_size)
    
    def get_sentence_vector(self, sentence_tokens):
        vectors = [self.get_word_vector(word) for word in sentence_tokens]
        valid_vectors = [v for v in vectors if np.any(v)]
        
        if valid_vectors:
            return np.mean(valid_vectors, axis=0)
        return np.zeros(self.vector_size)
    
    def find_similar_words(self, word, topn=10):
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []

# Usage example
sentences = [
    ['machine', 'learning', 'is', 'fascinating'],
    ['deep', 'learning', 'neural', 'networks'],
    ['natural', 'language', 'processing', 'nlp'],
    ['artificial', 'intelligence', 'ai', 'future']
]

w2v = Word2VecEmbeddings()
model = w2v.train(sentences)

# Get word vector
vector = w2v.get_word_vector('learning')
print(f"Vector for 'learning': {vector[:5]}...")

# Find similar words
similar = w2v.find_similar_words('learning')
print(f"Similar to 'learning': {similar}")
```

### Pre-trained Embeddings
```python
import gensim.downloader as api
from transformers import AutoTokenizer, AutoModel
import torch

class PretrainedEmbeddings:
    def __init__(self, model_name='word2vec-google-news-300'):
        if 'word2vec' in model_name:
            self.model = api.load(model_name)
            self.embedding_type = 'word2vec'
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.embedding_type = 'transformer'
    
    def get_embeddings(self, texts):
        if self.embedding_type == 'word2vec':
            return self._get_word2vec_embeddings(texts)
        else:
            return self._get_transformer_embeddings(texts)
    
    def _get_word2vec_embeddings(self, texts):
        embeddings = []
        for text in texts:
            words = text.split()
            word_vectors = []
            for word in words:
                if word in self.model:
                    word_vectors.append(self.model[word])
            
            if word_vectors:
                embeddings.append(np.mean(word_vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.model.vector_size))
        
        return np.array(embeddings)
    
    def _get_transformer_embeddings(self, texts):
        embeddings = []
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', 
                                  truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
                embeddings.append(embedding.numpy())
        
        return np.array(embeddings)

# Usage
embeddings = PretrainedEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
texts = ["I love machine learning", "Natural language processing is fun"]
vectors = embeddings.get_embeddings(texts)
```

## Deep Learning for NLP

### LSTM for Text Classification
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = [self.vocab.get(word, 0) for word in text.split()]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices.extend([0] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices), torch.tensor(label)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
                 n_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, 
                           dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use the last hidden state
        output = self.dropout(hidden[-1])
        output = self.fc(output)
        
        return output

# Training function
def train_lstm_model(model, train_loader, val_loader, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        val_accuracy = 0
        val_total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_accuracy += (predicted == labels).sum().item()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}, '
              f'Val Accuracy: {val_accuracy/val_total:.4f}')
```

## Transformer Models

### BERT for Text Classification
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset

class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    
    def prepare_data(self, texts, labels, max_length=128):
        return BERTDataset(texts, labels, self.tokenizer, max_length)
    
    def train(self, train_dataset, val_dataset, output_dir='./results'):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
        return trainer
    
    def predict(self, texts):
        self.model.eval()
        predictions = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions.append(prediction.cpu().numpy())
        
        return predictions

# Usage example
bert_classifier = BERTClassifier()

# Prepare data
train_texts = ["I love this product", "This is terrible"]
train_labels = [1, 0]
train_dataset = bert_classifier.prepare_data(train_texts, train_labels)

# Train model
trainer = bert_classifier.train(train_dataset, train_dataset)

# Make predictions
predictions = bert_classifier.predict(["This is amazing!"])
```

### Custom Transformer Implementation
```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear layer
        output = self.W_o(attn_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, 
                 d_ff, max_seq_length, num_classes, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_seq_length, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def create_positional_encoding(self, max_seq_length, d_model):
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x, mask=None):
        seq_length = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :seq_length, :].to(x.device)
        x = self.dropout(x)
        
        # Transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Global average pooling and classification
        x = x.mean(dim=1)
        output = self.classifier(x)
        
        return output
```

## Advanced NLP Applications

### Named Entity Recognition
```python
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

class NERTrainer:
    def __init__(self, model_name='en_core_web_sm'):
        self.nlp = spacy.load(model_name)
        
        # Add NER component if not present
        if 'ner' not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe('ner')
        else:
            ner = self.nlp.get_pipe('ner')
        
        self.ner = ner
    
    def add_labels(self, labels):
        for label in labels:
            self.ner.add_label(label)
    
    def train(self, training_data, n_iter=100):
        # training_data format: [(text, {"entities": [(start, end, label)]})]
        
        # Disable other pipes during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            
            for iteration in range(n_iter):
                losses = {}
                batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
                
                for batch in batches:
                    examples = []
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        examples.append(example)
                    
                    self.nlp.update(examples, drop=0.5, losses=losses)
                
                print(f"Iteration {iteration}, Losses: {losses}")
    
    def predict(self, text):
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_, ent.start_char, ent.end_char) 
                   for ent in doc.ents]
        return entities

# Usage example
training_data = [
    ("Apple Inc. is based in Cupertino.", 
     {"entities": [(0, 10, "ORG"), (23, 32, "GPE")]}),
    ("John works at Google in Mountain View.", 
     {"entities": [(0, 4, "PERSON"), (14, 20, "ORG"), (24, 37, "GPE")]})
]

ner_trainer = NERTrainer()
ner_trainer.add_labels(["ORG", "PERSON", "GPE"])
ner_trainer.train(training_data, n_iter=20)

# Make predictions
entities = ner_trainer.predict("Microsoft was founded by Bill Gates.")
print(entities)
```

### Text Summarization
```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TextSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.summarizer = pipeline('summarization', 
                                 model=self.model, 
                                 tokenizer=self.tokenizer)
    
    def summarize(self, text, max_length=150, min_length=50):
        # Handle long texts by chunking
        max_chunk_length = 1024  # BART's max input length
        
        if len(text.split()) <= max_chunk_length:
            summary = self.summarizer(text, 
                                    max_length=max_length, 
                                    min_length=min_length, 
                                    do_sample=False)
            return summary[0]['summary_text']
        else:
            # Split text into chunks and summarize each
            chunks = self._split_text(text, max_chunk_length)
            summaries = []
            
            for chunk in chunks:
                summary = self.summarizer(chunk, 
                                        max_length=max_length//len(chunks), 
                                        min_length=min_length//len(chunks), 
                                        do_sample=False)
                summaries.append(summary[0]['summary_text'])
            
            # Combine and summarize again
            combined_summary = ' '.join(summaries)
            final_summary = self.summarizer(combined_summary, 
                                          max_length=max_length, 
                                          min_length=min_length, 
                                          do_sample=False)
            return final_summary[0]['summary_text']
    
    def _split_text(self, text, max_length):
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

# Usage
summarizer = TextSummarizer()
long_text = """
Your long article text here...
"""
summary = summarizer.summarize(long_text)
print(summary)
```

## Model Evaluation and Deployment

### Evaluation Metrics for NLP
```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class NLPEvaluator:
    def __init__(self):
        pass
    
    def evaluate_classification(self, y_true, y_pred, labels=None):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Detailed classification report
        if labels:
            report = classification_report(y_true, y_pred, target_names=labels)
            print("\nDetailed Classification Report:")
            print(report)
        
        return results
    
    def evaluate_ner(self, true_entities, pred_entities):
        # Entity-level evaluation
        true_set = set(true_entities)
        pred_set = set(pred_entities)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def evaluate_text_generation(self, generated_texts, reference_texts):
        # BLEU score for text generation
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        bleu_scores = []
        smoothing = SmoothingFunction().method1
        
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            gen_tokens = gen_text.split()
            ref_tokens = [ref_text.split()]  # BLEU expects list of reference lists
            
            bleu = sentence_bleu(ref_tokens, gen_tokens, smoothing_function=smoothing)
            bleu_scores.append(bleu)
        
        avg_bleu = np.mean(bleu_scores)
        return {'bleu_score': avg_bleu}
```

## Best Practices and Tips

### 1. Data Quality and Preprocessing
- **Clean your data**: Remove noise, handle encoding issues
- **Consistent preprocessing**: Apply same preprocessing to train/test data
- **Handle class imbalance**: Use appropriate sampling techniques
- **Validate your preprocessing**: Check results manually

### 2. Model Selection and Training
- **Start simple**: Begin with traditional ML before deep learning
- **Use pre-trained models**: Leverage transfer learning when possible
- **Proper validation**: Use appropriate train/validation/test splits
- **Monitor overfitting**: Use early stopping and regularization

### 3. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Example hyperparameter tuning pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

parameters = {
    'tfidf__max_features': [1000, 5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'classifier__alpha': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

### 4. Production Deployment
```python
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load trained model and preprocessor
model = joblib.load('nlp_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Vectorize
        features = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].max()
        
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(probability),
            'text': text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
```

## Conclusion

Natural Language Processing has evolved from rule-based systems to sophisticated neural networks capable of understanding and generating human-like text. The field continues to advance rapidly with new architectures and techniques.

Key takeaways:
- **Preprocessing is crucial**: Clean, consistent data preprocessing significantly impacts performance
- **Start with pre-trained models**: Leverage existing models and fine-tune for your specific task
- **Evaluation matters**: Use appropriate metrics and validation strategies
- **Consider computational resources**: Balance model complexity with available resources
- **Stay updated**: The field evolves rapidly with new techniques and models

Whether you're building a chatbot, sentiment analyzer, or document classifier, understanding these fundamental concepts and techniques will help you build effective NLP systems.