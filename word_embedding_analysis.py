import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec, FastText
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re
from sklearn.manifold import TSNE
from collections import Counter
import requests
import os
import zipfile
import io
import time
from tqdm import tqdm

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')


class WordEmbeddingAnalysis:
    def __init__(self, data_dir='data'):
        """Initialize the analysis pipeline."""
        self.data_dir = data_dir
        self.stop_words = set(stopwords.words('english'))
        self.models = {}

        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def download_datasets(self):
        """Download IMDB and SemEval datasets."""
        # IMDB Dataset
        imdb_url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

        if not os.path.exists(f"{self.data_dir}/aclImdb"):
            print("Downloading IMDB dataset...")
            try:
                response = requests.get(imdb_url, stream=True)
                response.raise_for_status()

                # Extract tar.gz file
                import tarfile
                tar_file = tarfile.open(fileobj=io.BytesIO(response.content))
                tar_file.extractall(path=self.data_dir)
                tar_file.close()
                print("IMDB dataset downloaded and extracted successfully.")
            except Exception as e:
                print(f"Error downloading IMDB dataset: {e}")
        else:
            print("IMDB dataset already exists.")

        # SemEval-2017 Task 4 - Updated URL
        # Note: Direct link may not work; using alternative approach
        print("SemEval dataset: Please download manually from the SemEval 2017 website")
        print("and place the files in the data/semeval2017 directory")

        # Create directory for SemEval even if download fails
        if not os.path.exists(f"{self.data_dir}/semeval2017"):
            os.makedirs(f"{self.data_dir}/semeval2017")

    def load_imdb_data(self):
        """Load and preprocess IMDB dataset."""
        print("Loading IMDB dataset...")
        imdb_data = []

        # Load positive and negative reviews
        for sentiment in ['pos', 'neg']:
            path = f"{self.data_dir}/aclImdb/train/{sentiment}"
            if os.path.exists(path):
                files = os.listdir(path)
                # Use a smaller subset for quicker processing
                files = files[:5000] if len(files) > 5000 else files

                for filename in files:
                    if filename.endswith('.txt'):
                        with open(os.path.join(path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                            text = f.read()
                            imdb_data.append({
                                'text': text,
                                'sentiment': 1 if sentiment == 'pos' else 0
                            })

        # Convert to DataFrame
        self.imdb_df = pd.DataFrame(imdb_data)
        print(f"IMDB dataset loaded: {len(self.imdb_df)} reviews")
        return self.imdb_df

    def load_semeval_data(self):
        """Load and preprocess SemEval-2017 dataset."""
        print("Loading SemEval-2017 dataset...")
        semeval_path = f"{self.data_dir}/semeval2017/twitter-2016train-A.txt"

        # If the SemEval dataset isn't available, create a small dummy dataset
        if not os.path.exists(semeval_path):
            print("SemEval-2017 dataset file not found. Creating dummy dataset for testing.")
            dummy_data = [
                {'id': 1, 'sentiment': 1, 'text': 'I love this product! It works great.'},
                {'id': 2, 'sentiment': 0, 'text': 'The product is okay, nothing special.'},
                {'id': 3, 'sentiment': -1, 'text': 'Terrible experience with this product. Avoid!'},
                {'id': 4, 'sentiment': 1, 'text': 'Amazing service and quality. Highly recommended!'},
                {'id': 5, 'sentiment': 0, 'text': 'It does the job but could be better.'}
            ]
            self.semeval_df = pd.DataFrame(dummy_data)
            print(f"Created dummy SemEval dataset: {len(self.semeval_df)} tweets")
            return self.semeval_df

        try:
            self.semeval_df = pd.read_csv(
                semeval_path,
                sep='\t',
                header=None,
                names=['id', 'sentiment', 'text'],
                encoding='utf-8',
                quoting=3,  # QUOTE_NONE
                error_bad_lines=False
            )

            # Convert sentiment labels to numeric
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            self.semeval_df['sentiment'] = self.semeval_df['sentiment'].map(sentiment_map)

            print(f"SemEval dataset loaded: {len(self.semeval_df)} tweets")
            return self.semeval_df
        except Exception as e:
            print(f"Error loading SemEval dataset: {e}")
            # Create dummy data
            dummy_data = [
                {'id': 1, 'sentiment': 1, 'text': 'I love this product! It works great.'},
                {'id': 2, 'sentiment': 0, 'text': 'The product is okay, nothing special.'},
                {'id': 3, 'sentiment': -1, 'text': 'Terrible experience with this product. Avoid!'},
                {'id': 4, 'sentiment': 1, 'text': 'Amazing service and quality. Highly recommended!'},
                {'id': 5, 'sentiment': 0, 'text': 'It does the job but could be better.'}
            ]
            self.semeval_df = pd.DataFrame(dummy_data)
            print(f"Created dummy SemEval dataset: {len(self.semeval_df)} tweets")
            return self.semeval_df

    def preprocess_text(self, text):
        """Preprocess text by removing special characters, lowercasing, and tokenizing."""
        if not isinstance(text, str):
            text = str(text)

        # Convert to lowercase
        text = text.lower()

        # Remove URLs, mentions, and hashtags (for Twitter data)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)

        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]

        return tokens

    def preprocess_data(self, df):
        """Preprocess the dataset."""
        print("Preprocessing data...")
        df['tokens'] = df['text'].apply(self.preprocess_text)
        return df

    def train_word2vec(self, sentences, model_type='cbow', vector_size=100, window=5, min_count=5, epochs=5):
        """Train Word2Vec model with the given sentences."""
        print(f"Training Word2Vec ({model_type}) model...")
        sg = 0 if model_type == 'cbow' else 1
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            sg=sg,
            epochs=epochs
        )
        return model

    def train_fasttext(self, sentences, vector_size=100, window=5, min_count=5, epochs=5):
        """Train FastText model with the given sentences."""
        print("Training FastText model...")
        model = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4,
            epochs=epochs
        )
        return model

    def load_glove(self, vector_size=100):
        """Load pre-trained GloVe embeddings."""
        print("Loading GloVe embeddings...")
        glove_path = f"{self.data_dir}/glove.6B.{vector_size}d.txt"
        word2vec_output_path = f"{self.data_dir}/glove.6B.{vector_size}d.word2vec.txt"

        # Check if GloVe embeddings exist
        if not os.path.exists(glove_path):
            # Updated GloVe URL
            glove_url = f"https://nlp.stanford.edu/data/glove.6B.zip"
            print(f"GloVe embeddings not found. Please download them manually from:")
            print(f"{glove_url}")
            print(f"Extract and place the file 'glove.6B.{vector_size}d.txt' in the '{self.data_dir}' directory.")

            # Create a simple mock GloVe model for demonstration
            print("Creating mock GloVe embeddings for demonstration...")
            mock_glove = {}
            words = ["good", "bad", "happy", "sad", "love", "hate", "excellent", "terrible",
                     "amazing", "awful", "movie", "film", "actor", "review", "positive", "negative"]

            for word in words:
                mock_glove[word] = np.random.normal(size=vector_size)

            # Create a KeyedVectors object
            model = KeyedVectors(vector_size=vector_size)
            for word, vector in mock_glove.items():
                model.add_vectors([word], [vector])

            return model

        # Convert GloVe format to Word2Vec format if needed
        if not os.path.exists(word2vec_output_path):
            try:
                glove2word2vec(glove_path, word2vec_output_path)
            except Exception as e:
                print(f"Error converting GloVe to Word2Vec format: {e}")
                return None

        # Load the converted embeddings
        try:
            model = KeyedVectors.load_word2vec_format(word2vec_output_path)
            return model
        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            return None

    def build_custom_embeddings(self, all_sentences):
        """Build and train different word embedding models."""
        # Train Word2Vec (CBOW)
        self.models['word2vec_cbow'] = self.train_word2vec(
            all_sentences, model_type='cbow', vector_size=100, window=5, min_count=2
        )

        # Train Word2Vec (Skip-gram)
        self.models['word2vec_skipgram'] = self.train_word2vec(
            all_sentences, model_type='skipgram', vector_size=100, window=5, min_count=2
        )

        # Train FastText
        self.models['fasttext'] = self.train_fasttext(
            all_sentences, vector_size=100, window=5, min_count=2
        )

        # Load GloVe
        glove_model = self.load_glove(vector_size=100)
        if glove_model:
            self.models['glove'] = glove_model

        return self.models

    def find_most_similar_words(self, model, word, n=10):
        """Find the n most similar words to the given word."""
        try:
            if hasattr(model, 'wv'):
                similar_words = model.wv.most_similar(word, topn=n)
            else:
                similar_words = model.most_similar(word, topn=n)
            return similar_words
        except KeyError:
            print(f"Word '{word}' not in vocabulary")
            return []
        except Exception as e:
            print(f"Error finding similar words for '{word}': {e}")
            return []

    def analyze_word_similarity(self, sentiment_words=['good', 'bad', 'happy', 'sad', 'excellent', 'terrible']):
        """Analyze word similarity across all models."""
        similarity_results = {}

        for model_name, model in self.models.items():
            print(f"Analyzing word similarity for {model_name}...")
            model_results = {}

            for word in sentiment_words:
                try:
                    if hasattr(model, 'wv'):
                        similar_words = model.wv.most_similar(word, topn=10)
                    else:
                        similar_words = model.most_similar(word, topn=10)
                    model_results[word] = similar_words
                except KeyError:
                    print(f"Word '{word}' not in vocabulary for {model_name}")
                    model_results[word] = []
                except Exception as e:
                    print(f"Error analyzing '{word}' for {model_name}: {e}")
                    model_results[word] = []

            similarity_results[model_name] = model_results

        return similarity_results

    def compute_word_pair_similarity(self, word_pairs):
        """Compute cosine similarity between word pairs across all models."""
        similarity_scores = {}

        for model_name, model in self.models.items():
            print(f"Computing word pair similarity for {model_name}...")
            model_scores = {}

            for word1, word2 in word_pairs:
                try:
                    if hasattr(model, 'wv'):
                        if word1 in model.wv and word2 in model.wv:
                            score = model.wv.similarity(word1, word2)
                            model_scores[f"{word1}-{word2}"] = score
                        else:
                            print(f"One of the words '{word1}' or '{word2}' not in vocabulary for {model_name}")
                            model_scores[f"{word1}-{word2}"] = None
                    else:
                        if word1 in model and word2 in model:
                            score = model.similarity(word1, word2)
                            model_scores[f"{word1}-{word2}"] = score
                        else:
                            print(f"One of the words '{word1}' or '{word2}' not in vocabulary for {model_name}")
                            model_scores[f"{word1}-{word2}"] = None
                except Exception as e:
                    print(f"Error computing similarity for '{word1}-{word2}' in {model_name}: {e}")
                    model_scores[f"{word1}-{word2}"] = None

            similarity_scores[model_name] = model_scores

        return similarity_scores

    def visualize_embeddings(self, words_to_plot, filename='embedding_visualization.png'):
        """Visualize word embeddings using t-SNE."""
        print("Visualizing embeddings...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = axes.flatten()

        for i, (model_name, model) in enumerate(self.models.items()):
            # Get word vectors
            word_vectors = []
            valid_words = []

            for word in words_to_plot:
                try:
                    if hasattr(model, 'wv'):
                        if word in model.wv:
                            vector = model.wv[word]
                            word_vectors.append(vector)
                            valid_words.append(word)
                    else:
                        if word in model:
                            vector = model[word]
                            word_vectors.append(vector)
                            valid_words.append(word)
                except Exception as e:
                    print(f"Error getting vector for '{word}' in {model_name}: {e}")

            # Check if we have enough words
            if len(word_vectors) < 2:
                axes[i].text(0.5, 0.5, f"Not enough words in vocabulary for {model_name}",
                             ha='center', va='center', fontsize=14)
                axes[i].set_title(model_name)
                continue

            # Convert to numpy array
            word_vectors = np.array(word_vectors)

            # Apply t-SNE
            try:
                perplexity = min(30, max(2, len(word_vectors) - 1))
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                reduced_vectors = tsne.fit_transform(word_vectors)

                # Plot
                axes[i].scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])

                # Add word labels
                for j, word in enumerate(valid_words):
                    axes[i].annotate(word, (reduced_vectors[j, 0], reduced_vectors[j, 1]))

                axes[i].set_title(model_name)
            except Exception as e:
                print(f"Error visualizing embeddings for {model_name}: {e}")
                axes[i].text(0.5, 0.5, f"Error visualizing {model_name}: {str(e)}",
                             ha='center', va='center', fontsize=12, wrap=True)
                axes[i].set_title(model_name)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        print(f"Visualization saved as {filename}")

    def run_pipeline(self):
        """Run the complete analysis pipeline."""
        # Download datasets
        self.download_datasets()

        # Load and preprocess IMDB dataset
        imdb_df = self.load_imdb_data()
        imdb_df = self.preprocess_data(imdb_df)

        # Load and preprocess SemEval dataset
        semeval_df = self.load_semeval_data()
        if not semeval_df.empty:
            semeval_df = self.preprocess_data(semeval_df)

        # Combine sentences from both datasets
        all_sentences = list(imdb_df['tokens'])
        if not semeval_df.empty:
            all_sentences.extend(list(semeval_df['tokens']))

        # Build word embeddings
        self.build_custom_embeddings(all_sentences)

        # Define sentiment words for analysis
        sentiment_words = [
            'good', 'bad', 'happy', 'sad', 'love', 'hate',
            'excellent', 'terrible', 'amazing', 'awful'
        ]

        # Analyze word similarity
        similarity_results = self.analyze_word_similarity(sentiment_words)

        # Define word pairs for similarity comparison
        word_pairs = [
            ('good', 'great'), ('bad', 'terrible'),
            ('happy', 'joyful'), ('sad', 'depressed'),
            ('love', 'adore'), ('hate', 'despise'),
            ('movie', 'film'), ('actor', 'actress'),
            ('positive', 'negative'), ('like', 'dislike')
        ]

        # Compute similarity between word pairs
        similarity_scores = self.compute_word_pair_similarity(word_pairs)

        # Visualize embeddings - use words we know are in the vocabulary
        words_to_plot = ['good', 'bad', 'happy', 'sad', 'love', 'hate', 'movie', 'film']
        self.visualize_embeddings(words_to_plot)

        return {
            'similarity_results': similarity_results,
            'similarity_scores': similarity_scores
        }

    def compare_models(self, similarity_results, similarity_scores):
        """Compare different embedding models based on similarity results."""
        print("\n--- Model Comparison ---")

        # Compare how models represent sentiment words
        for word in list(similarity_results.get('word2vec_cbow', {}).keys()):
            print(f"\nTop similar words for '{word}':")

            for model_name in similarity_results.keys():
                if model_name in similarity_results and word in similarity_results[model_name]:
                    similar_words = similarity_results[model_name].get(word, [])
                    if similar_words:
                        similar_word_list = [w for w, _ in similar_words[:5]]
                        print(f"  {model_name}: {', '.join(similar_word_list)}")

        # Compare similarity scores
        print("\nSimilarity scores across models:")

        # Create a DataFrame for easier comparison
        scores_data = {}

        # Make sure we have data from at least one model
        if 'word2vec_cbow' in similarity_scores and similarity_scores['word2vec_cbow']:
            for pair in similarity_scores['word2vec_cbow'].keys():
                pair_scores = {}
                for model_name in similarity_scores.keys():
                    if model_name in similarity_scores and pair in similarity_scores[model_name]:
                        pair_scores[model_name] = similarity_scores[model_name][pair]
                scores_data[pair] = pair_scores

        if scores_data:
            scores_df = pd.DataFrame(scores_data).T
            print(scores_df)

            # Save comparison to CSV
            scores_df.to_csv('model_comparison.csv')
            print("Comparison saved to model_comparison.csv")

            return scores_df
        else:
            print("No valid comparison data available.")
            return pd.DataFrame()


# Run the analysis
if __name__ == "__main__":
    analyzer = WordEmbeddingAnalysis()
    results = analyzer.run_pipeline()
    comparison_df = analyzer.compare_models(
        results['similarity_results'],
        results['similarity_scores']
    )