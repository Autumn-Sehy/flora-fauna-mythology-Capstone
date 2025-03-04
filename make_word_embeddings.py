#!/usr/bin/env python3
import os
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from config import Config


class CbowModel(nn.Module):
    """
    This is for making CBOW embeddings (continuous bag of words - which was my halloween costume
    last year)
    """

    def __init__(self, vocab_size, embedding_dim):
        super(CbowModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words):
        embeds = self.embeddings(context_words)
        hidden = torch.mean(embeds, dim=1)
        output = self.linear(hidden)
        return output


class WordEmbeddingMaker:
    """
    Make embeddings for the stories themselves - no pretrained models!
    This way I can do topic modeling, look up flora/fauna, etc. without
    bias from GloVe or Word2Vec. And I can do T-SNE visualizations
    which are honestly one of the coolest ways to see the data 🦋🌱
    """

    def __init__(self, config=None):
        self.config = config or Config
        self.device = self.config.DEVICE
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.corpus = []
        self.model = None
        self.embeddings = {}
        self.embedding_dir = os.path.join(self.config.PICKLE_DIR, "embeddings")
        os.makedirs(self.embedding_dir, exist_ok=True)
        self.checkpoints_dir = os.path.join(self.embedding_dir, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.embedding_config = {
            "EMBEDDING_DIM": 100,
            "CONTEXT_SIZE": 2,
            "MIN_WORD_COUNT": 5,
            "EPOCHS": 5,
            "BATCH_SIZE": 256,
            "LEARNING_RATE": 0.001
        }

    def load_stories(self, stories_dir=None):
        """Load stories from directory && tokenize them"""
        stories_dir = stories_dir or self.config.DATA_PATH
        print(f"Loading stories from {stories_dir}")

        corpus = []
        total_stories = sum(len(files) for _, _, files in os.walk(stories_dir))

        with tqdm(total=total_stories, desc="Loading stories", unit="story") as pbar:
            for root, _, files in os.walk(stories_dir):
                for file in files:
                    file_path = os.path.join(root, file)

                    # I didn't always properly encode stories when I did OCR
                    for encoding in ['utf-8', 'ISO-8859-1', 'Windows-1252', 'utf-16']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                text = f.read().strip()
                                tokens = word_tokenize(text.lower())
                                if len(tokens) > 0:
                                    corpus.extend(tokens)
                                break
                        except UnicodeDecodeError:
                            continue

                    pbar.update(1)

        print(f"🌸 Loaded {len(corpus)} tokens from {total_stories} stories")
        return corpus

    def build_vocab(self, min_count=None):
        """
        Build vocab from myths
        """
        min_count = min_count or self.embedding_config["MIN_WORD_COUNT"]
        print("🦎 Building vocabulary...")

        word_counts = Counter(self.corpus)

        # In myths words are often uncommon but important
        # Filter out super short words instead of low frequency ones
        filtered_words = [word for word, count in word_counts.items() if len(word) > 1]

        #word mappings
        self.word_to_idx = {word: i + 1 for i, word in enumerate(filtered_words)}
        self.word_to_idx['<UNK>'] = 0

        # Reverse mapping for later lookup
        self.idx_to_word = {i: word for word, i in self.word_to_idx.items()}

        print(f"Vocab built!")
        return self.word_to_idx

    def make_training_data(self, context_size=None):
        """
        Create CBOW training data - context words predict center word
        """
        context_size = context_size or self.embedding_config["CONTEXT_SIZE"]
        print("🦢 Creating training data...")

        training_data = []
        valid_indices = [i for i, word in enumerate(self.corpus)
                         if word in self.word_to_idx]

        for i in tqdm(valid_indices, desc="Preparing batches"):
            center_word = self.corpus[i]

            context_indices = list(range(max(0, i - context_size), i)) + \
                              list(range(i + 1, min(len(self.corpus), i + context_size + 1)))

            context_words = [self.corpus[idx] for idx in context_indices
                             if self.corpus[idx] in self.word_to_idx]

            if context_words:
                training_data.append((context_words, center_word))

        print(f"🪷 Created {len(training_data)} training examples")
        return training_data

    def train(self, training_data, embedding_dim=None, epochs=None, batch_size=None, learning_rate=None):
        """
        Train the model, train train the model.
        """
        embedding_dim = embedding_dim or self.embedding_config["EMBEDDING_DIM"]
        epochs = epochs or self.embedding_config["EPOCHS"]
        batch_size = batch_size or self.embedding_config["BATCH_SIZE"]
        learning_rate = learning_rate or self.embedding_config["LEARNING_RATE"]

        print(f"Training on {self.device} (yay for my gaming laptop)")

        vocab_size = len(self.word_to_idx)
        model = CbowModel(vocab_size, embedding_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()
        losses = []

        random.shuffle(training_data)
        num_batches = (len(training_data) + batch_size - 1) // batch_size

        for epoch in range(epochs):
            total_loss = 0

            for i in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs}"):
                batch = training_data[i * batch_size:(i + 1) * batch_size]

                context_tensors = []
                target_tensors = []

                for context_words, target_word in batch:
                    context_indices = [self.word_to_idx.get(w, 0) for w in context_words]
                    context_size = self.embedding_config["CONTEXT_SIZE"]

                    if len(context_indices) < 2 * context_size:
                        context_indices.extend([0] * (2 * context_size - len(context_indices)))
                    elif len(context_indices) > 2 * context_size:
                        context_indices = context_indices[:2 * context_size]

                    target_idx = self.word_to_idx.get(target_word, 0)

                    context_tensors.append(context_indices)
                    target_tensors.append(target_idx)

                context_tensor = torch.LongTensor(context_tensors).to(self.device)
                target_tensor = torch.LongTensor(target_tensors).to(self.device)

                # Forward pass
                model.zero_grad()
                log_probs = model(context_tensor)

                # Calculate loss && update
                loss = loss_function(log_probs, target_tensor)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

            self.save_checkpoint(model, epoch)

        print(f"🌻 Training done! Final loss is: {losses[-1]:.4f}")
        self.model = model
        return model, losses

    def get_embeddings(self):
        """Extract word embeddings from trained model"""
        print("Extracting embeddings 🦔 🦔 🦔 ")

        if self.model is None:
            raise ValueError("Model has not been trained yet - whoooooops")

        embeddings = {}
        with torch.no_grad():
            # Get the embedding matrix
            embedding_matrix = self.model.embeddings.weight.cpu().numpy()

            # Map words to their embeddings
            for word, idx in tqdm(self.word_to_idx.items(), desc="Building embeddings dictionary"):
                embeddings[word] = embedding_matrix[idx].tolist()

        self.embeddings = embeddings
        return embeddings

    def save_embeddings(self, output_path=None):
        """Save embeddings to file"""
        pickle_path = output_path
        if pickle_path is None:
            pickle_path = self.config.get_pickle_path("word_embeddings")

        print(f"🦜 Saving embeddings to {pickle_path}")

        # Make sure directory exists
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)

        # Save as pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.embeddings, f)

        print(f"🌹 Embeddings pickled & saved!")
        return pickle_path

    def save_checkpoint(self, model, epoch):
        """Save checkpoint to resume training later"""
        checkpoint_path = os.path.join(self.checkpoints_dir, f"cbow_epoch_{epoch + 1}.pt")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'config': {
                'embedding_dim': self.embedding_config["EMBEDDING_DIM"],
                'context_size': self.embedding_config["CONTEXT_SIZE"]
            }
        }

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint"""
        print(f"🦩 Looooooooooooooaddddinggggggggggg....")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        config = checkpoint['config']

        self.word_to_idx = checkpoint['word_to_idx']
        self.idx_to_word = checkpoint['idx_to_word']

        vocab_size = len(self.word_to_idx)
        self.model = CbowModel(vocab_size, config['embedding_dim']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print("🌼 Checkpoint loaded & ready to roll")
        return self.model

    def run(self, embedding_dim=None, context_size=None, min_word_count=None,
            epochs=None, batch_size=None, learning_rate=None):
        """Run the embedding creation process"""
        print("🦋 Word Embeddings Maker 🦋")
        print(f"Using device: {self.device} - let's hope it's fast!")

        # Update embedding configuration if provided
        if embedding_dim:
            self.embedding_config["EMBEDDING_DIM"] = embedding_dim
        if context_size:
            self.embedding_config["CONTEXT_SIZE"] = context_size
        if min_word_count:
            self.embedding_config["MIN_WORD_COUNT"] = min_word_count
        if epochs:
            self.embedding_config["EPOCHS"] = epochs
        if batch_size:
            self.embedding_config["BATCH_SIZE"] = batch_size
        if learning_rate:
            self.embedding_config["LEARNING_RATE"] = learning_rate

        print("🦁 Starting the embedding adventures...")
        self.corpus = self.load_stories()
        self.build_vocab()
        training_data = self.make_training_data()
        self.train(training_data)

        self.get_embeddings()

        embedding_path = self.save_embeddings()

        print("\n🦄🌸 Word embeddings created! 🌸🦄")
        return embedding_path

