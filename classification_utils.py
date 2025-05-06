import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class SentimentClassifier:
    def __init__(self, embedding_models):
        """Initialize classifier with embedding models."""
        self.embedding_models = embedding_models
        self.classifiers = {}
        self.results = {}

    def create_document_vectors(self, tokens, model, model_name):
        """Create document vectors by averaging word embeddings."""
        doc_vectors = []

        for token_list in tqdm(tokens, desc=f"Creating document vectors for {model_name}"):
            if not token_list:
                # Empty document
                vector_size = 100  # Default vector size
                if model_name == 'glove':
                    vector_size = model.vector_size
                elif hasattr(model, 'wv') and hasattr(model.wv, 'vector_size'):
                    vector_size = model.wv.vector_size

                doc_vectors.append(np.zeros(vector_size))
                continue

            word_vectors = []
            for token in token_list:
                try:
                    if model_name == 'glove':
                        word_vectors.append(model[token])
                    else:
                        word_vectors.append(model.wv[token])
                except KeyError:
                    # Skip words not in vocabulary
                    continue

            if word_vectors:
                doc_vectors.append(np.mean(word_vectors, axis=0))
            else:
                # No word was in vocabulary
                vector_size = 100  # Default
                if model_name == 'glove':
                    vector_size = model.vector_size
                elif hasattr(model, 'wv') and hasattr(model.wv, 'vector_size'):
                    vector_size = model.wv.vector_size

                doc_vectors.append(np.zeros(vector_size))

        return np.array(doc_vectors)

    def train_classifier(self, X_train, y_train, classifier_type='logistic'):
        """Train a classifier on the embeddings."""
        if classifier_type == 'logistic':
            classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif classifier_type == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

        classifier.fit(X_train, y_train)
        return classifier

    def evaluate_classifier(self, classifier, X_test, y_test):
        """Evaluate the classifier performance."""
        y_pred = classifier.predict(X_test)

        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }

        return results

    def evaluate_embedding_models(self, tokens, labels, test_size=0.2, classifier_type='logistic'):
        """Evaluate all embedding models on sentiment classification task."""
        for model_name, model in self.embedding_models.items():
            print(f"\nEvaluating {model_name}...")

            # Create document vectors
            doc_vectors = self.create_document_vectors(tokens, model, model_name)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                doc_vectors, labels, test_size=test_size, random_state=42
            )

            # Train classifier
            classifier = self.train_classifier(X_train, y_train, classifier_type)
            self.classifiers[model_name] = classifier

            # Evaluate
            results = self.evaluate_classifier(classifier, X_test, y_test)
            self.results[model_name] = results

            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"F1 Score: {results['f1']:.4f