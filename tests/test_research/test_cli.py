"""The research CLI is a lazy dispatcher, independent of training dependencies."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import research


def test_top_level_help_loads_no_builder(monkeypatch, capsys):
    monkeypatch.setattr(research, "import_module", lambda _: pytest.fail("Eager builder import"))
    with pytest.raises(SystemExit) as result:
        research.main(["--help"])
    assert result.value.code == 0
    assert "bgc-source" in capsys.readouterr().out


def test_dispatch_passes_one_shared_parser_and_explicit_arguments(monkeypatch):
    calls = []

    def configure(parser):
        parser.add_argument("--sample", type=int, required=True)

    def run(args, parser):
        calls.append((args.command, args.sample, parser.prog))
        return 7

    def load(name):
        assert name == "src.research.builders.cr4"
        calls.append(name)
        return SimpleNamespace(configure_parser=configure, run=run)

    monkeypatch.setattr(research, "import_module", load)
    assert research.main(["cr4", "--sample", "3"]) == 7
    assert calls == ["src.research.builders.cr4", ("cr4", 3, "research.py cr4")]


def test_qualified_module_hook_has_the_same_contract(monkeypatch):
    monkeypatch.setattr(research, "COMMANDS", {"review": ("src.research.review_app", "Review")})
    imported = []

    def load(name):
        imported.append(name)
        return SimpleNamespace(configure_parser=lambda parser: None, run=lambda args, parser: 0)

    monkeypatch.setattr(research, "import_module", load)
    assert research.main(["review"]) == 0
    assert imported == ["src.research.review_app"]


def test_source_help_works_outside_repository_without_model_imports(tmp_path):
    entry = Path(research.__file__).resolve()
    code = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('research_cli', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.main(['cr4', '--help'])
except SystemExit as result:
    assert result.code == 0
assert not {'torch', 'transformers', 'numpy', 'pyarrow'} & sys.modules.keys()
assert 'src.research.builders.arxiv' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(entry)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--candidate-dir" in result.stdout
