import importlib
import json
import os
import pathlib
import sys
import threading
import types

import pytest

# Ensure project root is importable regardless of test runner cwd
ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("LOG_DIR", str(LOG_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def stub_external_sdks():
    """Provide stub implementations for third-party SDKs used at import time."""
    # Use real ddtrace if installed (CI with Test Visibility), otherwise stub
    _ddtrace_installed = "ddtrace" in sys.modules or importlib.util.find_spec("ddtrace") is not None
    if _ddtrace_installed:
        import ddtrace as ddtrace_module
        try:
            import ddtrace.llmobs as llmobs_module
        except ImportError:
            llmobs_module = types.ModuleType("ddtrace.llmobs")
        tracer_module = types.ModuleType("ddtrace.tracer")
        tracer_module.trace = ddtrace_module.tracer.trace
    else:
        ddtrace_module = types.ModuleType("ddtrace")
        llmobs_module = types.ModuleType("ddtrace.llmobs")
        tracer_module = types.ModuleType("ddtrace.tracer")

    class _FakeAnnotationContext:
        """Stub for LLMObs.annotation_context() context manager."""

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    class _FakeManagedPrompt:
        """Stub for ManagedPrompt returned by LLMObs.get_prompt()."""

        def __init__(self, prompt_id, template, label=None):
            self.id = prompt_id
            self.version = "test"
            self.label = label or "production"
            self._template = template

        def format(self, **variables):
            if isinstance(self._template, str):
                result = self._template
                for key, value in variables.items():
                    result = result.replace("{{" + key + "}}", str(value))
                return result
            if isinstance(self._template, list):
                result = []
                for msg in self._template:
                    content = msg["content"]
                    for key, value in variables.items():
                        content = content.replace("{{" + key + "}}", str(value))
                    result.append({"role": msg["role"], "content": content})
                return result
            return self._template

        def to_annotation_dict(self, **variables):
            return {
                "id": self.id,
                "version": self.version,
                "template": self._template,
                "variables": variables,
            }

    class _FakeLLMObs:
        calls = []

        @classmethod
        def enable(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))

        @classmethod
        def annotation_context(cls, **kwargs):
            """Return a no-op context manager."""
            return _FakeAnnotationContext(**kwargs)

        @classmethod
        def get_prompt(cls, prompt_id, label=None, fallback=None):
            return _FakeManagedPrompt(prompt_id, fallback, label=label)

        @classmethod
        def clear_prompt_cache(cls, **kwargs):
            pass

        @classmethod
        def refresh_prompt(cls, prompt_id, label=None):
            return _FakeManagedPrompt(prompt_id, [], label=label)

    class _FakeSpan:
        trace_id = 0
        span_id = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def set_tag(self, *args, **kwargs):
            return None

    class _FakeContextProvider:
        def activate(self, *args, **kwargs):
            pass

    class _FakeTracer:
        context_provider = _FakeContextProvider()
        _root_span = None

        def trace(self, *args, **kwargs):
            return _FakeSpan()

        def current_root_span(self):
            return self._root_span

    llmobs_module.LLMObs = _FakeLLMObs
    ddtrace_module.llmobs = llmobs_module
    if not _ddtrace_installed:
        fake_tracer = _FakeTracer()
        ddtrace_module.tracer = fake_tracer
        tracer_module.trace = fake_tracer.trace
    sys.modules["ddtrace"] = ddtrace_module
    sys.modules["ddtrace.llmobs"] = llmobs_module
    sys.modules["ddtrace.tracer"] = tracer_module

    # Stub openai.OpenAI client -- Responses API shape
    openai_module = types.ModuleType("openai")

    class _FakeResponses:
        def __init__(self):
            self._responses = []
            self._should_raise = False
            self.calls = 0
            self._lock = threading.Lock()

        def queue_response(self, content):
            self._responses.append(content)

        def raise_on_create(self):
            self._should_raise = True

        def create(self, *_, **_kwargs):
            with self._lock:
                self.calls += 1
                if self._should_raise:
                    raise Exception("forced failure")
                content = self._responses.pop(0) if self._responses else "yes"
                if isinstance(content, dict):
                    content = json.dumps(content)
                return types.SimpleNamespace(output_text=content)

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):
            self.responses = _FakeResponses()

    openai_module.OpenAI = _FakeOpenAI
    openai_module.AsyncOpenAI = _FakeOpenAI
    sys.modules["openai"] = openai_module

    # Stub the openai-agents SDK so chatbot.py can import without the full
    # openai type tree.  Only the symbols actually used by chatbot.py are
    # provided; tests that exercise the agent itself should patch
    # chatbot.generate_chat_response directly.
    agents_module = types.ModuleType("agents")

    def _noop_decorator(fn=None, **kwargs):
        """Stand-in for @function_tool -- returns the function unchanged."""
        if fn is not None:
            return fn
        return lambda f: f

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeRunner:
        @staticmethod
        def run_sync(*args, **kwargs):
            return types.SimpleNamespace(final_output="stub", new_items=[])

    agents_module.Agent = _FakeAgent
    agents_module.Runner = _FakeRunner
    agents_module.function_tool = _noop_decorator

    agents_items_module = types.ModuleType("agents.items")
    agents_items_module.ToolCallOutputItem = type("ToolCallOutputItem", (), {})
    agents_module.items = agents_items_module

    sys.modules["agents"] = agents_module
    sys.modules["agents.items"] = agents_items_module

    yield

    # Ensure stubs cleaned up for any downstream imports
    sys.modules.pop("ddtrace", None)
    sys.modules.pop("ddtrace.llmobs", None)
    sys.modules.pop("ddtrace.tracer", None)
    sys.modules.pop("openai", None)
    sys.modules.pop("agents", None)
    sys.modules.pop("agents.items", None)


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Point the database module at an isolated SQLite file and initialize schema."""
    db_file = tmp_path / "startups.db"
    monkeypatch.setenv("DEVTOOLS_DB_PATH", str(db_file))
    monkeypatch.setenv("DEVTOOLS_DATA_DIR", str(tmp_path))

    import database

    importlib.reload(database)
    database.init_db()
    return database


@pytest.fixture
def reset_ai_classifier(monkeypatch):
    """Reload ai_classifier with stubbed dependencies and provide easy client control."""
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_CACHE", "1")
    monkeypatch.setenv("AI_CLASSIFIER_DISABLE_BATCH", "1")
    monkeypatch.setenv("AI_CLASSIFIER_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import ai_classifier

    importlib.reload(ai_classifier)
    ai_classifier._get_openai_client()
    yield ai_classifier
