"""Build a discovery dataset for the HuggingFace Space demo.

This script samples from the already-filtered training data (processed by
download_data.py), runs model inference to generate summaries/topics/emotions,
and uploads the result to HuggingFace Datasets.

Data sources (no news articles — the model is trained on papers & books):
  - ArXiv academic papers (summarization training data)
  - Project Gutenberg / Goodreads literary works (summarization training data)
  - GoEmotions social media text (emotion training data, for emotion diversity)
  - Curated technical blog posts (hand-written, covering all 7 topic categories)

The training data has already been filtered by download_data.py for:
  - English content only
  - Quality text (no metadata, errata, technical manuals)
  - No Shakespeare/plays (excluded titles)
  - Proper book descriptions (from Goodreads, not plot summaries)
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.inference.factory import create_inference_pipeline  # noqa: E402

# --------------- Curated Content ---------------

# Hand-written blog posts covering underrepresented topics so the demo
# has content across all 7 topic categories beyond just paper/book data.
CURATED_ENTRIES: list[dict[str, Any]] = [
    # ── Technology ──
    {
        "id": "blog_tech_0",
        "title": "Understanding Transformer Attention Mechanisms",
        "text": (
            "The transformer architecture, introduced in 'Attention Is All You Need' by "
            "Vaswani et al. (2017), revolutionized natural language processing by replacing "
            "recurrence with self-attention mechanisms. The key innovation is the scaled "
            "dot-product attention, which computes attention weights as softmax(QK^T/sqrt(d_k))V, "
            "where Q, K, and V are query, key, and value matrices respectively. Multi-head "
            "attention extends this by projecting inputs into multiple subspaces, allowing the "
            "model to attend to different types of relationships simultaneously. This "
            "parallelizable architecture enabled training on much larger datasets and led to "
            "breakthrough models like BERT, GPT, and T5. The self-attention mechanism has O(n^2) "
            "complexity with respect to sequence length, which has motivated research into "
            "efficient attention variants such as linear attention, sparse attention, and flash "
            "attention."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "The transformer architecture uses multi-head self-attention to replace recurrence, "
            "enabling parallelized training and leading to advances like BERT, GPT, and T5."
        ),
    },
    {
        "id": "blog_tech_1",
        "title": "The Rise of Retrieval-Augmented Generation",
        "text": (
            "Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for "
            "combining large language models with external knowledge sources. Rather than "
            "relying solely on parametric knowledge stored in model weights, RAG systems first "
            "retrieve relevant documents from a knowledge base using dense retrieval methods, "
            "then feed these documents as context to the language model for generation. This "
            "approach addresses several key limitations of pure generative models: hallucination, "
            "outdated information, and lack of domain-specific knowledge. Modern RAG systems "
            "typically use embedding models to encode both queries and documents into a shared "
            "vector space, enabling efficient similarity search through approximate nearest "
            "neighbor algorithms. The retrieved passages are then prepended to the user query "
            "before being processed by the language model, allowing it to ground its responses "
            "in factual content."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Retrieval-Augmented Generation retrieves relevant documents before generation, "
            "grounding LLM outputs in factual content and addressing hallucination."
        ),
    },
    {
        "id": "blog_tech_2",
        "title": "Mixture of Experts: Scaling Language Models Efficiently",
        "text": (
            "Mixture of Experts (MoE) architectures have become a key scaling strategy for "
            "modern language models. Instead of routing every token through all parameters, MoE "
            "models use a gating network to selectively activate only a subset of expert networks "
            "for each input. This allows models to have dramatically more total parameters while "
            "keeping compute costs manageable. Google's Switch Transformer demonstrated that MoE "
            "could scale to trillion-parameter models, while Mixtral from Mistral AI showed that "
            "open-weight MoE models could match or exceed much larger dense models. The main "
            "challenges include load balancing across experts, communication overhead in "
            "distributed settings, and the large memory footprint from storing all expert weights. "
            "Recent innovations like expert parallelism and capacity factors have helped address "
            "these practical deployment issues."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Mixture of Experts models use gating to activate expert subsets, enabling efficient "
            "scaling while managing compute and load balancing challenges."
        ),
    },
    {
        "id": "blog_tech_3",
        "title": "Fine-Tuning vs In-Context Learning: A Practical Guide",
        "text": (
            "The debate between fine-tuning and in-context learning represents a fundamental "
            "tension in modern AI deployment. Fine-tuning adapts a pre-trained model's weights "
            "through gradient updates on task-specific data, yielding strong performance but "
            "requiring compute, data curation, and management of multiple model versions. "
            "In-context learning, by contrast, leverages the model's existing capabilities by "
            "providing task examples directly in the prompt at inference time. Parameter-efficient "
            "methods like LoRA and QLoRA have lowered the barrier for fine-tuning by updating "
            "only a fraction of model weights. Meanwhile, advances in prompt engineering, "
            "chain-of-thought reasoning, and retrieval-augmented generation have expanded what "
            "in-context learning can achieve. In practice, the choice depends on task complexity, "
            "data availability, latency requirements, and cost constraints."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Fine-tuning and in-context learning offer complementary trade-offs: fine-tuning "
            "excels for stable high-volume tasks while in-context learning suits diverse settings."
        ),
    },
    # ── Business ──
    {
        "id": "blog_biz_0",
        "title": "The Economics of AI Compute: Cloud vs On-Premise",
        "text": (
            "As organizations scale their AI workloads, the economics of compute infrastructure "
            "become critical. Cloud GPU providers like AWS, Azure, and GCP offer flexibility and "
            "quick provisioning but at premium hourly rates that compound over continuous training "
            "runs. On-premise GPU clusters require significant capital expenditure but can achieve "
            "dramatically lower per-hour costs for sustained workloads. The breakeven analysis "
            "depends on GPU utilization rates, with organizations running GPUs above 60% "
            "utilization typically finding on-premise more cost-effective within 18-24 months. "
            "Hybrid approaches are emerging, where companies maintain on-premise capacity for "
            "baseline workloads and burst to cloud for peak demand. The rapid depreciation of GPU "
            "hardware and the accelerating pace of new chip releases add complexity to these "
            "calculations."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Organizations must weigh cloud flexibility against on-premise cost savings, with "
            "hybrid approaches emerging as GPU utilization and hardware depreciation shape decisions."
        ),
    },
    {
        "id": "blog_biz_1",
        "title": "AI Adoption in Enterprise: Barriers and Strategies",
        "text": (
            "Enterprise AI adoption continues to face significant barriers despite the "
            "technology's demonstrated potential. A 2024 McKinsey survey found that while 65% of "
            "organizations regularly use generative AI, only 15% have deployed it at scale across "
            "multiple business functions. Key barriers include data quality and integration "
            "challenges, skills gaps in the workforce, regulatory uncertainty, and difficulty "
            "measuring ROI. Successful adopters typically start with focused use cases that have "
            "clear metrics, build cross-functional teams combining domain and technical expertise, "
            "and invest in data infrastructure before scaling AI applications. Change management "
            "is often underestimated — employees need training, process redesign, and clear "
            "communication about how AI will augment rather than replace their roles."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Despite growing AI use, scaling remains challenging; organizations succeed by "
            "focusing on specific use cases, cross-functional teams, and robust data infrastructure."
        ),
    },
    {
        "id": "blog_biz_2",
        "title": "The Subscription Economy and Digital Transformation",
        "text": (
            "The shift from one-time purchases to subscription models has fundamentally reshaped "
            "business strategy across industries. Software-as-a-Service companies pioneered this "
            "transformation, but the model has since expanded to physical goods, media, healthcare, "
            "and even automotive features. Annual recurring revenue has become the primary "
            "valuation metric, replacing traditional measures like earnings per share. This shift "
            "changes everything from customer acquisition strategies to financial reporting. "
            "Companies must now focus on customer lifetime value, churn reduction, and net revenue "
            "retention rather than quarterly sales targets. The economics favor businesses that "
            "can reduce marginal costs while increasing switching costs through data lock-in and "
            "network effects."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Subscription models have transformed business strategy and valuation metrics across "
            "industries, shifting focus to customer lifetime value and recurring revenue."
        ),
    },
    # ── Philosophy ──
    {
        "id": "blog_phil_0",
        "title": "The Alignment Problem: Ethics of Artificial Intelligence",
        "text": (
            "The AI alignment problem asks how we can ensure that artificial intelligence systems "
            "behave in accordance with human values and intentions. As AI systems become more "
            "capable, the potential consequences of misalignment grow more severe. The challenge "
            "is multifaceted: human values are complex, context-dependent, and often "
            "contradictory; specifying reward functions that capture what we truly want is "
            "notoriously difficult; and advanced AI systems may find unexpected ways to satisfy "
            "their objectives that diverge from designer intent — a phenomenon known as reward "
            "hacking. Philosophical approaches to alignment draw from ethics, decision theory, "
            "and philosophy of mind. Consequentialist frameworks suggest optimizing for aggregate "
            "well-being, but face challenges in defining and measuring welfare across diverse "
            "populations. Deontological approaches propose encoding inviolable rules, but "
            "struggle with the rigidity of absolute constraints in novel situations."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "The alignment problem encompasses technical and philosophical challenges of ensuring "
            "AI systems reliably pursue human-compatible goals across ethical frameworks."
        ),
    },
    {
        "id": "blog_phil_1",
        "title": "Consciousness and Machine Sentience: Where Do We Draw the Line?",
        "text": (
            "The question of whether machines can be conscious has moved from philosophical "
            "speculation to urgent practical concern as AI systems display increasingly "
            "sophisticated behavior. The hard problem of consciousness — explaining why physical "
            "processes give rise to subjective experience — remains unsolved, making it impossible "
            "to definitively determine whether any system is conscious. Integrated Information "
            "Theory proposes that consciousness corresponds to integrated information (phi), "
            "suggesting that even simple systems might have minimal consciousness. Global "
            "Workspace Theory focuses on the broadcast of information across brain regions, which "
            "some argue could be replicated in artificial architectures. Functionalism holds that "
            "consciousness depends on functional organization rather than substrate, opening the "
            "door to machine consciousness."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Whether machines can be conscious remains unresolved, but competing theories have "
            "practical implications for our moral responsibilities toward AI systems."
        ),
    },
    {
        "id": "blog_phil_2",
        "title": "Free Will in the Age of Predictive Algorithms",
        "text": (
            "Predictive algorithms increasingly shape human behavior through recommendation "
            "systems, targeted advertising, and algorithmic curation of information. This raises "
            "profound questions about the nature of free will and autonomy. If an algorithm can "
            "predict with high accuracy what a person will choose, does this undermine the notion "
            "that the choice was free? Compatibilists argue that free will is compatible with "
            "determinism — what matters is that choices flow from our own desires and reasoning, "
            "regardless of predictability. Libertarian free will proponents contend that genuine "
            "agency requires the ability to have done otherwise, which algorithmic prediction "
            "seems to challenge. From a practical standpoint, the concern is less about "
            "metaphysical free will and more about manipulation: when platforms optimize for "
            "engagement, they can exploit cognitive biases to steer behavior in ways users would "
            "not reflectively endorse."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Algorithmic prediction raises questions about free will and autonomy, driving calls "
            "for transparency and regulation to protect cognitive liberty."
        ),
    },
    # ── History ──
    {
        "id": "blog_hist_0",
        "title": "The History of Computing: From Babbage to Transformers",
        "text": (
            "The history of computing spans nearly two centuries, from Charles Babbage's "
            "Analytical Engine in the 1830s to today's transformer-based AI systems. Key "
            "milestones include Alan Turing's formalization of computation in 1936, the "
            "construction of ENIAC in 1945, the invention of the transistor at Bell Labs in "
            "1947, and the development of integrated circuits in the late 1950s. The personal "
            "computer revolution of the 1980s, driven by Apple, IBM, and Microsoft, democratized "
            "computing. The World Wide Web, created by Tim Berners-Lee in 1989, transformed "
            "computing into a communication medium. Moore's Law — the observation that transistor "
            "density doubles roughly every two years — drove exponential growth in capability for "
            "decades. The deep learning revolution beginning around 2012, catalyzed by GPU "
            "parallelism and large datasets, has brought us to the current era of large language "
            "models and generative AI."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Computing evolved from Babbage's engine through Turing, transistors, PCs, the web, "
            "and deep learning — each transition transforming society and industry."
        ),
    },
    {
        "id": "blog_hist_1",
        "title": "The Printing Press and the Information Revolution",
        "text": (
            "Johannes Gutenberg's movable type printing press, developed around 1440, is widely "
            "considered one of the most transformative inventions in human history. Before the "
            "press, books were hand-copied by scribes, making them expensive and rare — a typical "
            "Bible required months of labor. Gutenberg's innovation reduced the cost of book "
            "production by roughly 80% within decades. By 1500, an estimated 20 million volumes "
            "had been printed across Europe. The consequences were profound: literacy rates "
            "increased, knowledge disseminated faster, the Protestant Reformation was fueled by "
            "printed pamphlets, the Scientific Revolution accelerated through the sharing of "
            "findings, and the concept of intellectual property emerged. The printing press also "
            "standardized languages, as printed texts established canonical spellings and grammar."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "The printing press transformed knowledge dissemination, fueling the Reformation and "
            "Scientific Revolution, with parallels to the modern internet age."
        ),
    },
    {
        "id": "blog_hist_2",
        "title": "The Space Race: Cold War Competition and Scientific Achievement",
        "text": (
            "The Space Race between the United States and the Soviet Union, spanning roughly from "
            "1957 to 1975, was one of the most dramatic technological competitions in history. "
            "The Soviet launch of Sputnik 1 in October 1957 shocked the Western world and "
            "catalyzed massive investment in science education and research. Yuri Gagarin's "
            "orbital flight in April 1961 further intensified the competition. President "
            "Kennedy's bold declaration that the US would land a man on the Moon before the "
            "decade's end mobilized unprecedented resources — NASA's budget peaked at 4.4% of "
            "federal spending in 1966. The Apollo 11 mission in July 1969 achieved that goal, "
            "with Neil Armstrong and Buzz Aldrin walking on the lunar surface. Beyond the "
            "geopolitical competition, the Space Race produced transformative technological "
            "spinoffs: satellite communications, weather forecasting, GPS, miniaturized "
            "electronics, and advances in materials science."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "The US-Soviet Space Race catalyzed unprecedented scientific investment, achieving "
            "the Moon landing and generating technological advances still in use today."
        ),
    },
    # ── Arts ──
    {
        "id": "blog_arts_0",
        "title": "Generative AI and the Future of Creative Expression",
        "text": (
            "The emergence of generative AI tools — Stable Diffusion, DALL-E, Midjourney for "
            "images, and ChatGPT, Claude for text — has ignited fierce debate about the nature "
            "of creativity and the future of artistic professions. Proponents argue these tools "
            "democratize creativity, enabling people without formal training to express visual "
            "and textual ideas. Critics counter that AI systems trained on existing art constitute "
            "a form of laundering creative labor, extracting value from artists' work without "
            "compensation or consent. The legal landscape is evolving rapidly: courts are "
            "grappling with questions of copyright for AI-generated works, while artists have "
            "filed class-action lawsuits against companies training on their art. From an "
            "aesthetic perspective, some argue that AI-generated art lacks intentionality and "
            "emotional depth, while others point to compelling collaborations between human "
            "artists and AI systems that produce genuinely novel forms."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "AI art tools raise questions about creativity, copyright, and artistic labor, with "
            "the impact depending on their role as replacements versus augmentations."
        ),
    },
    {
        "id": "blog_arts_1",
        "title": "The Renaissance: Art, Science, and Human Potential",
        "text": (
            "The European Renaissance, spanning roughly from the 14th to the 17th century, "
            "represented a profound cultural transformation rooted in the rediscovery of classical "
            "Greek and Roman thought. Beginning in the city-states of Italy — Florence, Venice, "
            "Rome — the movement produced extraordinary achievements in art, architecture, "
            "literature, and science. Artists like Leonardo da Vinci, Michelangelo, and Raphael "
            "developed techniques such as linear perspective, chiaroscuro, and anatomical accuracy "
            "that transformed visual representation. The Renaissance ideal of the 'universal man' "
            "encouraged polymathy: da Vinci was simultaneously an artist, engineer, anatomist, "
            "and inventor. The movement was supported by wealthy patrons like the Medici family "
            "and facilitated by the printing press, which accelerated the spread of ideas across "
            "Europe."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "The Renaissance revived classical learning, producing transformative art and science "
            "while establishing humanistic intellectual foundations for the modern world."
        ),
    },
    {
        "id": "blog_arts_2",
        "title": "Music and Emotion: The Neural Basis of Musical Experience",
        "text": (
            "Music's ability to evoke powerful emotions has fascinated researchers across "
            "neuroscience, psychology, and philosophy. Neuroimaging studies reveal that listening "
            "to music activates a distributed network of brain regions including the auditory "
            "cortex, limbic system, prefrontal cortex, and motor areas. Musical chills — the "
            "shivers experienced during particularly moving passages — are associated with "
            "dopamine release in the nucleus accumbens, the same reward circuit activated by "
            "food, sex, and drugs. Cross-cultural studies suggest some aspects of musical emotion "
            "perception are universal: major keys tend to be perceived as happy and minor keys as "
            "sad across cultures, though the strength of these associations varies. The BRECVEMA "
            "framework identifies eight mechanisms through which music induces emotions: brain "
            "stem reflexes, rhythmic entrainment, evaluative conditioning, emotional contagion, "
            "visual imagery, episodic memory, musical expectancy, and aesthetic judgment."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Musical emotion involves distributed brain activation and dopamine release, with the "
            "BRECVEMA framework explaining eight mechanisms of emotional response to music."
        ),
    },
    # ── Science (non-arxiv, to complement papers) ──
    {
        "id": "blog_sci_0",
        "title": "CRISPR and the Future of Genetic Medicine",
        "text": (
            "CRISPR-Cas9, a gene-editing technology adapted from bacterial immune systems, has "
            "transformed biological research and holds enormous therapeutic promise. The system "
            "uses a guide RNA to direct the Cas9 enzyme to a specific DNA sequence, where it "
            "makes a precise cut. The cell's repair mechanisms then either disable the gene or "
            "insert new genetic material. Since its development as a genome editing tool by "
            "Jennifer Doudna and Emmanuelle Charpentier in 2012, CRISPR has been used to model "
            "diseases, develop drought-resistant crops, and create gene drives for controlling "
            "disease-carrying mosquitoes. In medicine, clinical trials are underway for sickle "
            "cell disease, beta-thalassemia, certain cancers, and hereditary blindness. The first "
            "CRISPR-based therapy, Casgevy, was approved in 2023 for sickle cell disease."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "CRISPR gene editing has progressed from research tool to approved therapy, with "
            "ongoing clinical trials and ethical debates about germline modification."
        ),
    },
    {
        "id": "blog_sci_1",
        "title": "Dark Matter and Dark Energy: The Universe's Missing Pieces",
        "text": (
            "Observations over the past century have revealed that ordinary matter — the atoms "
            "that make up stars, planets, and people — accounts for only about 5% of the "
            "universe's total mass-energy content. Approximately 27% is dark matter, an invisible "
            "substance that interacts gravitationally but not electromagnetically, and about 68% "
            "is dark energy, a mysterious force driving the accelerating expansion of the "
            "universe. Evidence for dark matter includes galaxy rotation curves, gravitational "
            "lensing, and the cosmic microwave background. Despite decades of searching, dark "
            "matter particles have not been directly detected, leading some physicists to explore "
            "modified gravity theories as alternatives. Dark energy is even more mysterious: "
            "discovered through observations of distant supernovae in 1998, it behaves like "
            "Einstein's cosmological constant but its fundamental nature remains unknown."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Dark matter and dark energy constitute 95% of the universe but remain undetected "
            "directly, representing fundamental challenges in modern physics."
        ),
    },
    {
        "id": "blog_sci_2",
        "title": "Quantum Computing: Promise and Reality",
        "text": (
            "Quantum computing harnesses quantum mechanical phenomena — superposition, "
            "entanglement, and interference — to perform calculations that are intractable for "
            "classical computers. A quantum bit (qubit) can exist in a superposition of 0 and 1 "
            "simultaneously, and entangled qubits can represent exponentially many states. This "
            "enables quantum algorithms like Shor's (for integer factorization) and Grover's "
            "(for unstructured search) to achieve speedups over classical counterparts. However, "
            "practical quantum computing faces formidable engineering challenges: qubits are "
            "extremely fragile, requiring temperatures near absolute zero and isolation from "
            "environmental noise. Current devices are in the 'noisy intermediate-scale quantum' "
            "(NISQ) era, with hundreds of qubits but high error rates. Error correction schemes "
            "exist but require thousands of physical qubits per logical qubit."
        ),
        "source_type": "blog",
        "dataset": "curated",
        "reference_summary": (
            "Quantum computing promises exponential speedups but faces practical challenges; "
            "near-term applications likely include simulation and optimization."
        ),
    },
]


# --------------- Data Loading ---------------


def load_academic_papers(data_dir: Path, max_samples: int = 500) -> list[dict[str, Any]]:
    """Load academic paper samples from the summarization training data."""
    results: list[dict[str, Any]] = []

    for split in ["train", "test"]:
        summ_file = data_dir / "summarization" / f"{split}.jsonl"
        if not summ_file.exists():
            print(f"  Warning: {summ_file} not found")
            continue

        with open(summ_file) as f:
            for line in f:
                item = json.loads(line)
                if item.get("type") != "academic":
                    continue
                text = item.get("source", "")
                if len(text) < 500:
                    continue
                results.append(
                    {
                        "text": text[:2000],
                        "title": item.get("title", "Research Paper")[:150],
                        "reference_summary": item.get("summary", "")[:500],
                    }
                )

    random.shuffle(results)
    results = results[:max_samples]

    samples = []
    for i, item in enumerate(results):
        samples.append(
            {
                "id": f"paper_{i}",
                "title": item["title"],
                "text": item["text"],
                "source_type": "academic",
                "dataset": "arxiv",
                "reference_summary": item["reference_summary"],
            }
        )

    print(f"  Loaded {len(samples)} academic papers")
    return samples


def load_literary(data_dir: Path, max_samples: int = 500) -> list[dict[str, Any]]:
    """Load literary samples (Project Gutenberg / Goodreads) from training data."""
    literary: list[dict[str, Any]] = []
    seen_titles: set[str] = set()

    for split in ["train", "test"]:
        summ_file = data_dir / "summarization" / f"{split}.jsonl"
        if not summ_file.exists():
            print(f"  Warning: {summ_file} not found")
            continue

        with open(summ_file) as f:
            for line in f:
                item = json.loads(line)
                if item.get("type") != "literary":
                    continue
                title = item.get("title", "")
                if not title or title in seen_titles:
                    continue
                text = item.get("source", "")
                summary = item.get("summary", "")
                if len(text) < 300 or len(summary) < 50:
                    continue
                seen_titles.add(title)
                literary.append(
                    {
                        "text": text[:2000],
                        "title": title,
                        "reference_summary": summary[:600],
                    }
                )

    random.shuffle(literary)
    literary = literary[:max_samples]

    samples = []
    for i, item in enumerate(literary):
        samples.append(
            {
                "id": f"literary_{i}",
                "title": item["title"],
                "text": item["text"],
                "source_type": "literary",
                "dataset": "gutenberg",
                "reference_summary": item["reference_summary"],
            }
        )

    print(f"  Loaded {len(samples)} literary works (unique titles)")
    return samples


def load_emotion_texts(data_dir: Path, max_samples: int = 200) -> list[dict[str, Any]]:
    """Load emotion-labeled social media texts for emotion diversity.

    These are short GoEmotions texts.  They are NOT news — they come from
    Reddit comments labeled with one or more of 28 emotions.
    """
    emotion_items: list[dict[str, Any]] = []

    for split in ["test", "validation", "train"]:
        path = data_dir / "emotion" / f"{split}.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                text = item["text"].strip()
                emotions = item.get("emotions", [])
                if len(text) < 30 or not emotions:
                    continue
                emotion_items.append({"text": text, "emotions": emotions})

    # Sample for diversity: pick from across emotion categories
    by_emotion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in emotion_items:
        for e in item["emotions"]:
            by_emotion[e].append(item)

    seen_texts: set[str] = set()
    sampled: list[dict[str, Any]] = []
    for cat in sorted(by_emotion):
        candidates = by_emotion[cat]
        random.shuffle(candidates)
        for item in candidates:
            if item["text"] in seen_texts:
                continue
            seen_texts.add(item["text"])
            sampled.append(
                {
                    "id": f"emotion_{len(sampled)}",
                    "title": f"[{cat}] {item['text'][:60]}...",
                    "text": item["text"],
                    "source_type": "social",
                    "dataset": "goemotions",
                    "reference_summary": "",
                    "_ground_truth_emotion": cat,
                }
            )
            if len(sampled) >= max_samples:
                break
        if len(sampled) >= max_samples:
            break

    random.shuffle(sampled)
    print(f"  Loaded {len(sampled)} emotion-labeled social texts")
    return sampled


# --------------- Inference ---------------


def run_inference(pipeline: Any, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run model inference on all samples to get summaries, topics, and emotions."""
    results: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Running inference"):
        text = sample["text"]

        # Get model predictions
        summaries = pipeline.summarize([text])
        topics = pipeline.predict_topics([text])
        emotions = pipeline.predict_emotions([text])

        summary = summaries[0] if summaries else ""
        topic = topics[0] if topics else None
        emotion = emotions[0] if emotions else None

        # Primary emotion (highest confidence)
        primary_emotion = "neutral"
        emotion_confidence = 0.0
        if emotion and emotion.labels:
            primary_emotion = emotion.labels[0]
            emotion_confidence = emotion.scores[0]

        result = {
            "id": sample["id"],
            "title": sample["title"],
            "text": text,
            "source_type": sample["source_type"],
            "dataset": sample["dataset"],
            "topic": topic.label if topic else "Unknown",
            "topic_confidence": topic.confidence if topic else 0.0,
            "emotion": primary_emotion,
            "emotion_confidence": emotion_confidence,
            "generated_summary": summary,
            "reference_summary": sample.get("reference_summary", ""),
        }
        results.append(result)

    # Print distribution stats
    topic_dist: dict[str, int] = defaultdict(int)
    emotion_dist: dict[str, int] = defaultdict(int)
    for r in results:
        topic_dist[r["topic"]] += 1
        emotion_dist[r["emotion"]] += 1

    print(f"\nTopic distribution: {dict(topic_dist)}")
    print(f"Emotion distribution: {dict(emotion_dist)}")

    return results


# --------------- Main ---------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build discovery dataset for the HuggingFace Space demo"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--num-papers", type=int, default=500, help="Academic papers to sample")
    parser.add_argument("--num-literary", type=int, default=500, help="Literary works to sample")
    parser.add_argument(
        "--num-emotion", type=int, default=200, help="Emotion texts to sample (GoEmotions)"
    )
    parser.add_argument("--output", type=Path, default=Path("data/discovery_dataset.jsonl"))
    parser.add_argument("--push-to-hub", action="store_true", help="Push to HuggingFace Hub")
    parser.add_argument("--hub-repo", type=str, default="OliverPerrin/LexiMind-Discovery")
    args = parser.parse_args()

    random.seed(42)

    # ── Load data ──
    print("Loading data samples...")
    print("  Sources: ArXiv papers, Gutenberg/Goodreads books, GoEmotions, curated blogs")
    print("  (No news articles — model is trained on papers & books)\n")

    papers = load_academic_papers(args.data_dir, args.num_papers)
    literary = load_literary(args.data_dir, args.num_literary)
    emotion_texts = load_emotion_texts(args.data_dir, args.num_emotion)

    all_samples = papers + literary + emotion_texts + CURATED_ENTRIES
    random.shuffle(all_samples)

    print(
        f"\nTotal samples: {len(all_samples)}"
        f" ({len(papers)} papers, {len(literary)} literary,"
        f" {len(emotion_texts)} emotion, {len(CURATED_ENTRIES)} curated)"
    )

    if not all_samples:
        print("ERROR: No samples loaded! Check if data/processed exists and has data.")
        print("Run: python scripts/download_data.py --task summarization")
        return

    # ── Run model inference ──
    print(f"\nLoading model from {args.checkpoint}...")
    labels_path = Path("artifacts/labels.json")
    pipeline, _labels = create_inference_pipeline(
        args.checkpoint, labels_path, device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Running inference on all samples...")
    results = run_inference(pipeline, all_samples)

    # ── Save locally ──
    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for item in results:
            # Remove internal fields
            item.pop("_ground_truth_emotion", None)
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(results)} items")

    # ── Push to Hub ──
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.hub_repo}")
        # Re-read to ensure clean (no internal fields)
        clean: list[dict[str, Any]] = []
        with open(args.output) as f:
            for line in f:
                clean.append(json.loads(line))
        dataset = Dataset.from_list(clean)
        dataset.push_to_hub(
            args.hub_repo,
            private=False,
            commit_message=(
                f"Rebuild discovery dataset: {len(clean)} items "
                "(papers, books, emotion texts, curated blogs)"
            ),
        )
        print(f"Dataset available at: https://huggingface.co/datasets/{args.hub_repo}")

    print("\nDone!")


if __name__ == "__main__":
    main()
