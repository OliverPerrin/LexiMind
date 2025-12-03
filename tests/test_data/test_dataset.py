import json
import os
import tempfile
import unittest

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from src.data.dataset import (
    EmotionDataset,
    EmotionExample,
    SummarizationDataset,
    SummarizationExample,
    TopicDataset,
    TopicExample,
    load_emotion_jsonl,
    load_summarization_jsonl,
    load_topic_jsonl,
)


class TestDatasets(unittest.TestCase):
    def test_summarization_dataset(self):
        examples = [
            SummarizationExample(source="Source 1", summary="Summary 1"),
            SummarizationExample(source="Source 2", summary="Summary 2"),
        ]
        dataset = SummarizationDataset(examples)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], examples[0])
        self.assertEqual(dataset[1], examples[1])

    def test_emotion_dataset_auto_binarizer(self):
        examples = [
            EmotionExample(text="Text 1", emotions=["joy", "love"]),
            EmotionExample(text="Text 2", emotions=["sadness"]),
        ]
        dataset = EmotionDataset(examples)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], examples[0])
        self.assertTrue(hasattr(dataset, "binarizer"))
        self.assertIsInstance(dataset.binarizer, MultiLabelBinarizer)
        self.assertIn("joy", dataset.emotion_classes)
        self.assertIn("sadness", dataset.emotion_classes)

    def test_emotion_dataset_provided_binarizer(self):
        examples = [EmotionExample(text="Text 1", emotions=["joy"])]
        binarizer = MultiLabelBinarizer()
        binarizer.fit([["joy", "sadness"]])
        dataset = EmotionDataset(examples, binarizer=binarizer)
        self.assertEqual(dataset.binarizer, binarizer)
        self.assertEqual(set(dataset.emotion_classes), {"joy", "sadness"})

    def test_topic_dataset_auto_encoder(self):
        examples = [
            TopicExample(text="Text 1", topic="sports"),
            TopicExample(text="Text 2", topic="politics"),
        ]
        dataset = TopicDataset(examples)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], examples[0])
        self.assertTrue(hasattr(dataset, "encoder"))
        self.assertIsInstance(dataset.encoder, LabelEncoder)
        self.assertIn("sports", dataset.topic_classes)

    def test_topic_dataset_provided_encoder(self):
        examples = [TopicExample(text="Text 1", topic="sports")]
        encoder = LabelEncoder()
        encoder.fit(["sports", "tech"])
        dataset = TopicDataset(examples, encoder=encoder)
        self.assertEqual(dataset.encoder, encoder)
        self.assertEqual(set(dataset.topic_classes), {"sports", "tech"})


class TestDataLoading(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.jsonl_path = os.path.join(self.temp_dir.name, "data.jsonl")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_summarization_jsonl(self):
        data = [
            {"source": "S1", "summary": "Sum1"},
            {"source": "S2", "summary": "Sum2"},
        ]
        with open(self.jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        examples = load_summarization_jsonl(self.jsonl_path)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].source, "S1")
        self.assertEqual(examples[0].summary, "Sum1")

    def test_load_emotion_jsonl(self):
        data = [
            {"text": "T1", "emotions": ["e1"]},
            {"text": "T2", "emotions": ["e2", "e3"]},
        ]
        with open(self.jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        examples = load_emotion_jsonl(self.jsonl_path)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].text, "T1")
        self.assertEqual(examples[0].emotions, ["e1"])

    def test_load_topic_jsonl(self):
        data = [
            {"text": "T1", "topic": "top1"},
            {"text": "T2", "topic": "top2"},
        ]
        with open(self.jsonl_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

        examples = load_topic_jsonl(self.jsonl_path)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].text, "T1")
        self.assertEqual(examples[0].topic, "top1")

    def test_load_json_array(self):
        data = [
            {"source": "S1", "summary": "Sum1"},
            {"source": "S2", "summary": "Sum2"},
        ]
        with open(self.jsonl_path, "w") as f:
            json.dump(data, f)

        examples = load_summarization_jsonl(self.jsonl_path)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0].source, "S1")


if __name__ == "__main__":
    unittest.main()
