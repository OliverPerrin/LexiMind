import os
import json
from typing import Any, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Summarizer:
    def __init__(self, model_name: str = "t5-small", max_input: int = 512, max_output: int = 128, device: Optional[str] = None):
        self.model_name = model_name
        self.max_input = max_input
        self.max_output = max_output
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def load_data(self, split: str = "train", limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Load processed summarization data from JSON files.
        
        Args:
            split (str): Data split to load ('train', 'val', 'test')
            limit (int): Maximum number of samples to load (None for all)
            
        Returns:
            list: List of dictionaries with 'article' and 'summary' keys
        """
        # Resolve to project root regardless of current working directory
        root = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(root, "data", "processed", "summarization", f"{split}.json")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if limit:
            data = data[:limit]
        return data
        
    def encode(self, articles: List[str] | str, summaries: Optional[List[str] | str] = None):
        if isinstance(articles, str):
            articles = [articles]
        if summaries is not None and isinstance(summaries, str):
            summaries = [summaries]

        inputs = self.tokenizer(
            [f"summarize: {a}" for a in articles],
            max_length=self.max_input,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        result = {
            "input_ids": inputs.input_ids.to(self.device),
            "attention_mask": inputs.attention_mask.to(self.device)
        }

        if summaries is not None:
            labels = self.tokenizer(
                summaries,
                max_length=self.max_output,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).input_ids
            # Mask pad tokens in labels with -100 for loss
            labels[labels == self.tokenizer.pad_token_id] = -100
            result["labels"] = labels.to(self.device)
        return result

    def train(self, epochs: int = 3, batch_size: int = 4, train_limit: int = 2000, val_limit: int = 500, learning_rate: float = 5e-5):
        train_data = self.load_data("train", limit=train_limit)
        val_data = self.load_data("val", limit=val_limit)

        train_ds = _SummarizationDataset(train_data, self.tokenizer, self.max_input, self.max_output)
        val_ds = _SummarizationDataset(val_data, self.tokenizer, self.max_input, self.max_output) if val_data else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

        optim = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs} - {len(train_loader)} batches", flush=True)
            for i, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optim.step()
                optim.zero_grad()
                if (i % max(1, len(train_loader)//5 or 1)) == 0:
                    print(f"  step {i}/{len(train_loader)} - loss {float(loss):.4f}", flush=True)

            if val_loader:
                _ = self.evaluate(val_data[: min(100, len(val_data))])
        print("Training complete.", flush=True)

    def evaluate(self, val_data: List[Dict[str, str]]) -> float:
        if not val_data:
            return 0.0

        ds = _SummarizationDataset(val_data, self.tokenizer, self.max_input, self.max_output)
        loader = DataLoader(ds, batch_size=4)
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total += float(outputs.loss) * batch["input_ids"].size(0)
                count += batch["input_ids"].size(0)
        self.model.train()
        return total / max(count, 1)

    def summarize(self, text: str, max_length: Optional[int] = None, num_beams: int = 4) -> str:
        if not text.strip():
            return ""
        inputs = self.tokenizer(
            f"summarize: {text}",
            return_tensors="pt",
            max_length=self.max_input,
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length or self.max_output,
                num_beams=num_beams,
                length_penalty=2.0,
                early_stopping=True
            )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

    def save(self, path: str = "models/summarizer"):
        """
        Save the trained model and tokenizer.
        
        Args:
            path (str): Directory path to save the model
        """
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str = "models/summarizer"):
        """
        Load a pre-trained model from disk.
        
        Args:
            path (str): Directory path containing the saved model
            
        Returns:
            Summarizer: Loaded summarizer instance
        """
        obj = cls.__new__(cls)
        obj.model_name = path
        obj.max_input = 512
        obj.max_output = 128
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        obj.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obj.model.to(obj.device)
        return obj

class _SummarizationDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], tokenizer: Any, max_input: int, max_output: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_output = max_output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        inputs = self.tokenizer(
            f"summarize: {item['article']}",
            max_length=self.max_input,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = self.tokenizer(
            item["summary"],
            max_length=self.max_output,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

if __name__ == "__main__":
    print("Initializing summarizer...", flush=True)
    summarizer = Summarizer(model_name="t5-small")
    print("Starting a short training run...", flush=True)
    summarizer.train(epochs=3, batch_size=2, train_limit=100, val_limit=50)
    test_text = (
        "The quick brown fox jumps over the lazy dog. This is a common "
        "pangram used in typography and printing. It contains every letter of the "
        "alphabet at least once, making it useful for testing fonts and keyboards."
    )
    print("Generating summary...", flush=True)
    summary = summarizer.summarize(test_text)
    print(f"\nOriginal text: {test_text}")
    print(f"Summary: {summary}")
    summarizer.save()
