"""Fast regression tests that do not download models or datasets."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch


# The helpers under test import the Hugging Face symbol at module import time,
# but these unit tests deliberately never access the network.
if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = datasets_stub

from data.prepare_data import _code_only, _split_for_budget  # noqa: E402
from src.eval_code import _public_tests_from_prompt  # noqa: E402
from src.train import _supported_kwargs  # noqa: E402


class DataHelpersTest(unittest.TestCase):
    def test_code_only_uses_last_python_block(self):
        answer = "notes\n```python\nprint('old')\n```\n```python\ndef f():\n    return 2\n```"
        self.assertEqual(_code_only(answer), "def f():\n    return 2")

    def test_smoke_split_avoids_full_download(self):
        with patch.dict(os.environ, {"AGENTLIGHT_SMOKE": "1"}):
            self.assertEqual(_split_for_budget("train", 24), "train[:100]")
        with patch.dict(os.environ, {"AGENTLIGHT_SMOKE": ""}):
            self.assertEqual(_split_for_budget("train", 24), "train")


class EvaluationSafetyTest(unittest.TestCase):
    def test_only_visible_doctests_become_public_tests(self):
        prompt = '''def add(a, b):
    """
    >>> add(1, 2)
    3
    >>> print(add(2, 3))
    5
    """
'''
        tests = _public_tests_from_prompt(prompt)
        self.assertEqual(tests, ["assert (add(1, 2)) == (3)"])
        self.assertFalse(any("check(" in test for test in tests))


class CompatibilityTest(unittest.TestCase):
    def test_supported_kwargs_filters_api_drift(self):
        def old_api(model, tokenizer):
            return model, tokenizer

        values = {"model": 1, "tokenizer": 2, "processing_class": 3}
        self.assertEqual(
            _supported_kwargs(old_api, values),
            {"model": 1, "tokenizer": 2},
        )


if __name__ == "__main__":
    unittest.main()
