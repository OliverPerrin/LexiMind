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
                    
                # Saving as JSON, one file for each book
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
        
        # Process each CSV file separately (train.csv, validation.csv, test.csv)
        file_mapping = {
            'train.csv': 'train',
            'validation.csv': 'val',
            'test.csv': 'test'
        }
        
        for csv_file, split_name in file_mapping.items():
            file_path = os.path.join(input_folder, csv_file)
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue
                
            print(f"Processing {csv_file}...")
            df = pd.read_csv(file_path)
            
            # Check for required columns (article and highlights)
            if 'article' not in df.columns or 'highlights' not in df.columns:
                print(f"CSV {csv_file} must have 'article' and 'highlights' columns.")
                continue
                
            # Clean the text data
            df['article'] = df['article'].astype(str).apply(self.clean_text)
            df['summary'] = df['highlights'].astype(str).apply(self.clean_text)  # rename highlights to summary
            
            # Convert to records format
            records = df[['article', 'summary']].to_dict(orient='records')
            
            # Save as JSON
            output_file = os.path.join(output_folder, f"{split_name}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"Processed {csv_file}: {len(records)} samples saved to {split_name}.json")
            
        print("Summarization dataset processed and saved.")

    def process_emotion_dataset(self):
        """Process emotion dataset: clean, split, and save."""
        input_folder = "data/raw/emotion"
        output_folder = "data/processed/emotion"
        os.makedirs(output_folder, exist_ok=True)
        
        # Process each txt file (train.txt, val.txt, test.txt)
        for split_file in ['train.txt', 'val.txt', 'test.txt']:
            file_path = os.path.join(input_folder, split_file)
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue
                
            records = []
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and ';' in line:
                        # Split on the last semicolon to handle semicolons in text
                        text, label = line.rsplit(';', 1)
                        records.append({
                            'text': self.clean_text(text),
                            'label': label.strip()
                        })
            
            # Save as JSON
            split_name = split_file.replace('.txt', '')
            output_file = os.path.join(output_folder, f"{split_name}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"Processed {split_file}: {len(records)} samples saved to {split_name}.json")
        
        print("Emotion dataset processed and saved.")

    def process_topic_dataset(self):
        """Process topic dataset: clean, split, and save."""
        input_folder = "data/raw/topic"
        output_folder = "data/processed/topic"
        os.makedirs(output_folder, exist_ok=True)
        
        # Process each CSV file separately (train.csv, test.csv)
        file_mapping = {
            'train.csv': 'train',
            'test.csv': 'test'
        }
        
        # Class index to topic name mapping for AG News dataset
        class_map = {
            1: 'World',
            2: 'Sports', 
            3: 'Business',
            4: 'Science/Technology'
        }
        
        for csv_file, split_name in file_mapping.items():
            file_path = os.path.join(input_folder, csv_file)
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue
                
            print(f"Processing {csv_file}...")
            df = pd.read_csv(file_path)
            
            # Check for required columns
            if 'Class Index' not in df.columns:
                print(f"CSV {csv_file} must have 'Class Index' column.")
                continue
            
            # Concatenate title and description
            if 'Title' in df.columns and 'Description' in df.columns:
                text = df['Title'].astype(str) + ". " + df['Description'].astype(str)
            elif 'Title' in df.columns:
                text = df['Title'].astype(str)
            elif 'Description' in df.columns:
                text = df['Description'].astype(str)
            else:
                print("CSV must have 'Title' or 'Description' columns.")
                continue
                
            df['text'] = text.apply(self.clean_text)
            
            # Map numeric labels to category names
            df['label'] = df['Class Index'].map(class_map)
            
            # Convert to records format
            records = df[['text', 'label']].to_dict(orient='records')
            
            # Save as JSON
            output_file = os.path.join(output_folder, f"{split_name}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            print(f"Processed {csv_file}: {len(records)} samples saved to {split_name}.json")
        
        # Create validation split from training data
        if os.path.exists(os.path.join(output_folder, "train.json")):
            print("Creating validation split from training data...")
            with open(os.path.join(output_folder, "train.json"), "r", encoding="utf-8") as f:
                train_data = json.load(f)
            
            # Split training data into train and validation
            train_records, val_records = train_test_split(train_data, test_size=0.2, random_state=42)
            
            # Save updated train and new validation files
            with open(os.path.join(output_folder, "train.json"), "w", encoding="utf-8") as f:
                json.dump(train_records, f, ensure_ascii=False, indent=2)
            
            with open(os.path.join(output_folder, "val.json"), "w", encoding="utf-8") as f:
                json.dump(val_records, f, ensure_ascii=False, indent=2)
                
            print(f"Updated train.json: {len(train_records)} samples")
            print(f"Created val.json: {len(val_records)} samples")
            
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
