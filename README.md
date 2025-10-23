# LexiMind: Multi-Task Transformer for Document Analysis

A PyTorch-based multi-task learning system that performs abstractive summarization, emotion classification, and topic clustering on textual data using a shared Transformer encoder architecture.

## 🎯 Project Overview

LexiMind demonstrates multi-task learning (MTL) by training a single model to simultaneously:
1. **Abstractive Summarization**: Generate concise summaries with user-defined compression levels
2. **Emotion Classification**: Detect multiple emotions present in text (multi-label classification)
3. **Topic Clustering**: Group documents by semantic similarity for topic discovery

### Key Features
- Custom encoder-decoder Transformer architecture with shared representations
- Multi-task loss function with learnable task weighting
- Attention weight visualization for model interpretability
- Interactive web interface for real-time inference
- Trained on diverse corpora: news articles (CNN/DailyMail, BBC) and literary texts (Project Gutenberg)

## 🏗️ Architecture

```
Input Text
    ↓
┌─────────────────────┐
│  Shared Encoder     │  ← TransformerEncoder (6 layers)
│  (Multi-head Attn)  │
└─────────────────────┘
    ↓   ↓   ↓
    │   │   └──────────────┐
    │   │                  │
    │   └─────────┐        │
    │             │        │
    ↓             ↓        ↓
┌─────────┐  ┌────────┐  ┌─────────┐
│ Decoder │  │Classify│  │ Project │
│  Head   │  │  Head  │  │  Head   │
└─────────┘  └────────┘  └─────────┘
    ↓             ↓          ↓
Summary      Emotions    Embeddings
                          (for clustering)
```

## 📊 Datasets

- **CNN/DailyMail**: 300k+ news articles with human-written summaries
- **BBC News**: 2,225 articles across 5 categories
- **Project Gutenberg**: Classic literature for long-form text analysis

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/OliverPerrin/LexiMind.git
cd LexiMind
pip install -r requirements.txt
```

### Download Data
```bash
python src/download_datasets.py
```

### Train Model
```bash
python src/train.py --config configs/default.yaml
```

### Launch Interface
```bash
python src/app.py
```

## 📁 Project Structure

```
LexiMind/
├── src/
│   ├── models/
│   │   ├── encoder.py           # Shared Transformer encoder
│   │   ├── summarization.py     # Seq2seq decoder head
│   │   ├── emotion.py           # Multi-label classification head
│   │   └── clustering.py        # Projection head for embeddings
│   ├── data/
│   │   ├── download_datasets.py # Data acquisition
│   │   ├── preprocessing.py     # Text cleaning & tokenization
│   │   └── dataset.py           # PyTorch Dataset classes
│   ├── training/
│   │   ├── train.py             # Training loop
│   │   ├── losses.py            # Multi-task loss functions
│   │   └── metrics.py           # ROUGE, F1, silhouette scores
│   ├── inference/
│   │   └── pipeline.py          # End-to-end inference
│   ├── visualization/
│   │   └── attention.py         # Attention heatmap generation
│   └── app.py                   # Gradio/FastAPI interface
├── configs/
│   └── default.yaml             # Model & training hyperparameters
├── tests/
│   └── test_*.py                # Unit tests
├── notebooks/
│   └── exploratory.ipynb        # Data exploration & analysis
├── requirements.txt
└── README.md
```

## 🧪 Evaluation Metrics

| Task | Metric | Score |
|------|--------|-------|
| Summarization | ROUGE-1 / ROUGE-L | TBD |
| Emotion Classification | Macro F1 | TBD |
| Topic Clustering | Silhouette Score | TBD |

## 🔬 Technical Details

### Model Specifications
- **Encoder**: 6-layer Transformer (d_model=512, 8 attention heads)
- **Decoder**: 6-layer autoregressive Transformer
- **Vocab Size**: 32,000 (SentencePiece tokenizer)
- **Parameters**: ~60M total

### Training
- **Optimizer**: AdamW (lr=1e-4, weight_decay=0.01)
- **Scheduler**: Linear warmup (5000 steps) + cosine decay
- **Loss**: Weighted sum of cross-entropy (summarization), BCE (emotions), triplet loss (clustering)
- **Hardware**: Trained on single NVIDIA RTX 3090 (24GB VRAM)
- **Time**: ~48 hours for 10 epochs

### Multi-Task Learning Strategy
Uses uncertainty weighting ([Kendall et al., 2018](https://arxiv.org/abs/1705.07115)) to automatically balance task losses:

```
L_total = Σ (1/2σ²_i * L_i + log(σ_i))
```

where σ_i are learnable parameters representing task uncertainty.

## 🎨 Interface Preview

The web interface provides:
- Text input with real-time token count
- Compression level slider (20%-80%)
- Side-by-side original/summary comparison
- Emotion probability bars with color coding
- Interactive attention heatmap (click tokens to highlight attention)
- Downloadable results (JSON/CSV)

## 📈 Future Enhancements

- [ ] Add multilingual support (mBART)
- [ ] Implement beam search for better summaries
- [ ] Fine-tune on domain-specific corpora (medical, legal)
- [ ] Add semantic search across document embeddings
- [ ] Deploy as REST API with Docker
- [ ] Implement model distillation for mobile deployment

## 📚 References

- Vaswani et al. (2017) - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Lewis et al. (2019) - [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- Caruana (1997) - [Multitask Learning](https://link.springer.com/article/10.1023/A:1007379606734)
- Demszky et al. (2020) - [GoEmotions Dataset](https://arxiv.org/abs/2005.00547)

## 📄 License

GNU General Public License v3.0

## 👤 Author

**Oliver Perrin**
- Portfolio: [oliverperrin.com](https://oliverperrin.com)
- LinkedIn: [linkedin.com/in/oliverperrin](https://linkedin.com/in/oliverperrin)
- Email: oliver.t.perrin@gmail.com

---
