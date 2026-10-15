"""Automatic tests. They run WITHOUT internet, an API key or the embedding model:
a small word-count embedder and a fake Grok are used.   Run:  python tests.py"""
import json
import tempfile
import unittest
from pathlib import Path

import generate_sample_chapters
from embedder import HashingEmbedder
from grok_client import GrokClient, GrokError
from pdf_loader import chunk_text, load_and_chunk, chapter_name
from rag import StudyAssistant, ensure_index, NOT_FOUND
from vector_store import VectorStore


class FakeResponse:
    def __init__(self, status=200, content="fake answer [1]"):
        self.status_code, self._content, self.text = status, content, "error text"

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeGrok:
    """Records what would be sent to Grok."""
    def __init__(self):
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return "Fake answer from Grok [1]"


class StudyAssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.chapters, cls.store_dir = root / "chapters", root / "store"
        generate_sample_chapters.make(cls.chapters)
        cls.embedder = HashingEmbedder()
        cls.logs = []
        cls.store = ensure_index(cls.embedder, chapters_dir=cls.chapters, store_dir=cls.store_dir, log=cls.logs.append)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---------- loading and chunking ----------
    def test_chunk_text_respects_size_and_keeps_all_words(self):
        text = " ".join(f"This is sentence number {i} about learning." for i in range(60))
        chunks = chunk_text(text, size=300, overlap=60)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 300 + 60 for c in chunks))
        self.assertIn("sentence number 59", chunks[-1])                 # nothing is lost at the end
        self.assertTrue(set(chunks[0].split()[-3:]) & set(chunks[1].split()[:12]))   # neighbouring chunks overlap

    def test_loader_reads_five_chapters(self):
        chunks = load_and_chunk(self.chapters, log=lambda m: None)
        self.assertEqual(len({c["file"] for c in chunks}), 5)
        self.assertTrue(all(c["page"] >= 1 and c["text"] for c in chunks))
        self.assertEqual(chapter_name("Chapter_2_Machine_Learning_Basics.pdf"), "Chapter 2 Machine Learning Basics")

    def test_empty_folder_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                load_and_chunk(d)

    # ---------- saving embeddings ----------
    def test_embeddings_are_saved_and_reloaded_identically(self):
        self.assertTrue(VectorStore.exists(self.store_dir))
        loaded = VectorStore.load(self.store_dir)
        self.assertEqual(loaded.embeddings.shape, self.store.embeddings.shape)
        self.assertTrue((loaded.embeddings == self.store.embeddings).all())
        self.assertEqual(loaded.chunks, self.store.chunks)
        self.assertEqual(json.loads((self.store_dir / "index_meta.json").read_text())["n_chunks"], len(self.store.chunks))

    def test_saved_index_is_reused_and_rebuilt_when_pdfs_change(self):
        logs = []
        ensure_index(self.embedder, chapters_dir=self.chapters, store_dir=self.store_dir, log=logs.append)
        self.assertTrue(any("Loaded saved index" in m for m in logs))      # no re-embedding needed
        with tempfile.TemporaryDirectory() as d:                            # a copy where a PDF is added
            d = Path(d)
            ch, st = d / "c", d / "s"
            generate_sample_chapters.make(ch)
            ensure_index(self.embedder, chapters_dir=ch, store_dir=st, log=lambda m: None)
            (ch / "Chapter_6_extra.pdf").write_bytes((ch / "Chapter_1_What_is_AI.pdf").read_bytes())
            logs2 = []
            store2 = ensure_index(self.embedder, chapters_dir=ch, store_dir=st, log=logs2.append)
            self.assertTrue(any("changed" in m for m in logs2))
            self.assertEqual(len({c["file"] for c in store2.chunks}), 6)

    # ---------- retrieval ----------
    def test_retrieval_finds_the_right_chapter(self):
        assistant = StudyAssistant(self.embedder, self.store, FakeGrok(), min_score=0.05)
        cases = {
            "Who coined the term artificial intelligence?": "Chapter_1",
            "What is overfitting and how can it be reduced?": "Chapter_2",
            "What is backpropagation and gradient descent?": "Chapter_3",
            "What are the steps of the RAG pipeline?": "Chapter_4",
            "What is algorithmic bias and what causes it?": "Chapter_5",
        }
        for question, expected in cases.items():
            top = assistant.retrieve(question)[0]
            self.assertIn(expected, top["file"], f"{question!r} -> {top['file']}")
            self.assertGreater(top["score"], 0.05)

    # ---------- RAG answer ----------
    def test_prompt_contains_context_question_and_rules(self):
        grok = FakeGrok()
        assistant = StudyAssistant(self.embedder, self.store, grok, min_score=0.05)
        result = assistant.ask("What is overfitting?", history=[("hello", "hi")])
        self.assertEqual(result["answer"], "Fake answer from Grok [1]")
        self.assertTrue(result["grounded"] and 1 <= len(result["sources"]) <= 4)
        messages = grok.calls[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("ONLY", messages[0]["content"])
        self.assertEqual(messages[1:3], [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}])
        user = messages[-1]["content"]
        self.assertIn("QUESTION: What is overfitting?", user)
        self.assertIn("[1] (Chapter 2", user)
        self.assertIn("memorises the training data", user)                 # the retrieved text really is in the prompt

    def test_unrelated_question_does_not_call_grok(self):
        grok = FakeGrok()
        result = StudyAssistant(self.embedder, self.store, grok, min_score=0.05).ask("xyzzy plugh quux banana")
        self.assertEqual(result["answer"], NOT_FOUND)
        self.assertFalse(result["grounded"])
        self.assertEqual(grok.calls, [])

    def test_summary_and_quiz_use_their_own_instructions(self):
        grok = FakeGrok()
        assistant = StudyAssistant(self.embedder, self.store, grok, min_score=0.05)
        assistant.summarize("neural networks backpropagation")
        assistant.quiz("neural networks backpropagation", n=3)
        self.assertIn("study summary", grok.calls[0][0]["content"])
        self.assertIn("3 multiple-choice", grok.calls[1][0]["content"])

    # ---------- Grok client ----------
    def test_grok_client_request_format_and_errors(self):
        sent = {}

        def ok_post(url, headers, json, timeout):
            sent.update(url=url, headers=headers, json=json)
            return FakeResponse(200, "  hello  ")

        client = GrokClient(api_key="xai-test", model="grok-4", post=ok_post)
        self.assertEqual(client.chat([{"role": "user", "content": "hi"}]), "hello")
        self.assertEqual(sent["url"], "https://api.x.ai/v1/chat/completions")
        self.assertEqual(sent["headers"]["Authorization"], "Bearer xai-test")
        self.assertEqual(sent["json"]["model"], "grok-4")

        for status, word in ((401, "API key"), (404, "GROK_MODEL"), (429, "credits"), (500, "500")):
            c = GrokClient(api_key="k", post=lambda *a, status=status, **k: FakeResponse(status))
            with self.assertRaises(GrokError) as cm:
                c.chat([])
            self.assertIn(word, str(cm.exception))
        with self.assertRaises(GrokError):
            GrokClient(api_key="", post=ok_post).chat([])                    # missing key

        def broken_post(*a, **k):
            raise ConnectionError("no internet")
        with self.assertRaises(GrokError):
            GrokClient(api_key="k", post=broken_post).chat([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
