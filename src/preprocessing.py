# src/preprocessing.py
import re
import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class textPreprocessor:
    def __init__(self, max_length=512, model_name='bert-base-uncased'):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def clean_text(self, text: str) -> str:
        """Cleaning and Normalizing Text"""
        text = re.sub(r'\s+', ' ', text) # Getting rid of extra spaces
        text = re.sub(r'[^a-zA-Z0-9.,;:?!\'" ]', '', text) # Removing weird characters
        return text.strip()

    def tokenize_text(self, texts: list[str]):
        return self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
    def prepare_data(self, texts: list[str], labels=None):
        """Preparing Data for Training"""
        cleaned_texts = [self.clean_text(text) for text in texts]
        encoded = self.tokenize_text(cleaned_texts)
        
        if labels is not None:
            return encoded, tf.convert_to_tensor(labels)
        return encoded

    def load_books(self, folder_path="data/raw/books") -> list[str]:
        """Load books from text files in the specific folder"""
        texts = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors ="ignore") as file:
                    raw_text = file.read()
                    cleaned = self.clean_text(raw_text)
                    texts.append(cleaned)
        return texts

    def chunk_text(self, text: str, chunk_size=1000, overlap=100) -> list[str]:
        """Splits long texts into smaller segments or chunks"""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def save_preprocessed_books(self, data, input_folder="data/raw/books", output_folder="data/processed/books", chunk_size=1000, overlap=100):
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
                    cleaned = self.clean_text(raw_text)
                    chunks = self.chunk_text(cleaned, chunk_size, overlap)
                    
                # Save as JSON (one file for each book)
                out_file = os.path.join(output_folder, filename.replace(".txt", ".json"))
                with open(out_file, "w", encoding="utf-8") as out:
                    json.dump(chunks, out, ensure_ascii=False, indent=2)
                    
                print(f"Processed and saved {filename} → {out_file}")
                
                
    # ----- Dataset-specific processing methods ------

    def process_summarization_dataset(self):
        """Process summarization dataset: clean, split, and save."""
        input_folder = "data/raw/summarization/cnn_dailymail"
        output_folder = "data/processed/summarization"
        os.makedirs(output_folder, exist_ok=True)
        # Find first .csv file in input_folder
        csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
        if not csv_files:
            print(f"No CSV found in {input_folder}")
            return
        df = pd.read_csv(os.path.join(input_folder, csv_files[0]))
        # Expect columns: 'article', 'summary'
        if not {'article', 'summary'}.issubset(df.columns):
            print("CSV must have 'article' and 'summary' columns.")
            return
        df['article'] = df['article'].astype(str).apply(self.clean_text)
        df['summary'] = df['summary'].astype(str).apply(self.clean_text)
        train, temp = train_test_split(df, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        for split, data in zip(['train', 'val', 'test'], [train, val, test]):
            records = data[['article', 'summary']].to_dict(orient='records')
            with open(os.path.join(output_folder, f"{split}.json"), "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        print("Summarization dataset processed and saved.")

    def process_emotion_dataset(self):
        """Process emotion dataset: clean, split, and save."""
        input_folder = "data/raw/emotion"
        output_folder = "data/processed/emotion"
        os.makedirs(output_folder, exist_ok=True)
        csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
        if not csv_files:
            print(f"No CSV found in {input_folder}")
            return
        df = pd.read_csv(os.path.join(input_folder, csv_files[0]))
        # Expect columns: 'text', 'label'
        if not {'text', 'label'}.issubset(df.columns):
            print("CSV must have 'text' and 'label' columns.")
            return
        df['text'] = df['text'].astype(str).apply(self.clean_text)
        train, temp = train_test_split(df, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        for split, data in zip(['train', 'val', 'test'], [train, val, test]):
            records = data[['text', 'label']].to_dict(orient='records')
            with open(os.path.join(output_folder, f"{split}.json"), "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        print("Emotion dataset processed and saved.")

    def process_topic_dataset(self):
        """Process topic dataset: clean, split, and save."""
        input_folder = "data/raw/topic"
        output_folder = "data/processed/topic"
        os.makedirs(output_folder, exist_ok=True)
        csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
        if not csv_files:
            print(f"No CSV found in {input_folder}")
            return
        df = pd.read_csv(os.path.join(input_folder, csv_files[0]))
        # Expect at least 'title', 'description', 'label'
        if not 'label' in df.columns:
            print("CSV must have 'label' column.")
            return
        # Concatenate title and description if both exist
        if 'title' in df.columns and 'description' in df.columns:
            text = df['title'].astype(str) + ". " + df['description'].astype(str)
        elif 'title' in df.columns:
            text = df['title'].astype(str)
        elif 'description' in df.columns:
            text = df['description'].astype(str)
        else:
            print("CSV must have 'title' or 'description' columns.")
            return
        df['text'] = text.apply(self.clean_text)
        # Map numeric labels to strings if needed
        if pd.api.types.is_numeric_dtype(df['label']):
            # Try to find mapping if available
            label_map = None
            # Look for a column like 'label_name' or similar
            for col in df.columns:
                if col.lower() in ['label_name', 'topic', 'category']:
                    label_map = dict(zip(df['label'], df[col]))
                    break
            if label_map:
                df['label'] = df['label'].map(label_map)
            else:
                # Otherwise convert to a string
                df['label'] = df['label'].astype(str)
        train, temp = train_test_split(df, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        for split, data in zip(['train', 'val', 'test'], [train, val, test]):
            records = data[['text', 'label']].to_dict(orient='records')
            with open(os.path.join(output_folder, f"{split}.json"), "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        print("Topic dataset processed and saved.")


# ----- Main function for quick testing ------

if __name__ == "__main__":
    preprocessor = textPreprocessor(max_length=128)

    # Process and save all books
    preprocessor.save_preprocessed_books(data=None)

    # Load a processed book back
    import json
    with open("data/processed/books/pride_and_prejudice.json", "r") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from Pride and Prejudice")
    print(chunks[0][:200])  # printing first 200 chars of chunk

    # Process new datasets
    preprocessor.process_summarization_dataset()
    preprocessor.process_emotion_dataset()
    preprocessor.process_topic_dataset()
