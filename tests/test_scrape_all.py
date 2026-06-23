import types


class LoaderWrapper:
    def __init__(self, factory):
        self.factory = factory

    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        self.factory(module)


def configure_loader(monkeypatch, factories):
    import importlib.machinery
    import scrape_all

    def fake_spec_from_file_location(name, path):
        factory = factories[name]
        loader = LoaderWrapper(factory)
        return importlib.machinery.ModuleSpec(name, loader)

    def fake_module_from_spec(spec):
        module = spec.loader.create_module(spec)
        module.__spec__ = spec
        return module

    monkeypatch.setattr("scrape_all.importlib.util.spec_from_file_location", fake_spec_from_file_location)
    monkeypatch.setattr("scrape_all.importlib.util.module_from_spec", fake_module_from_spec)


def test_run_scraper_variants(monkeypatch):
    import scrape_all

    calls = []

    factories = {
        "scrape_github_trending": lambda module: module.__dict__.update(
            {"scrape_github_trending": lambda: calls.append("github")}
        ),
        "scrape_hackernews": lambda module: module.__dict__.update(
            {
                "scrape_hackernews": lambda: calls.append("hn-top"),
                "scrape_hackernews_show": lambda: calls.append("hn-show"),
            }
        ),
        "scrape_producthunt_api": lambda module: module.__dict__.update(
            {"scrape_producthunt_api": lambda: calls.append("ph")}
        ),
        "missing_main": lambda module: module.__dict__.update({"some_other": lambda: None}),
        "broken": lambda module: (_ for _ in ()).throw(RuntimeError("boom")),
    }

    configure_loader(monkeypatch, factories)

    assert scrape_all.run_scraper("scrape_github_trending", "GitHub")
    assert scrape_all.run_scraper("scrape_hackernews", "HN")
    assert scrape_all.run_scraper("scrape_producthunt_api", "PH")
    assert not scrape_all.run_scraper("missing_main", "No main")
    assert not scrape_all.run_scraper("broken", "Broken")

    assert calls == ["github", "hn-top", "hn-show", "ph"]


def test_scrape_all_main_records_results(monkeypatch):
    import scrape_all

    sequence = iter([True, False, True])
    monkeypatch.setattr("scrape_all.run_scraper", lambda name, desc: next(sequence))
    monkeypatch.setattr("scrape_all.init_db", lambda: None)
    recorded = []
    monkeypatch.setattr("scrape_all.record_scrape_completion", lambda summary: recorded.append(summary))

    scrape_all.main()
    # Scrapers 1 and 3 succeed (True, False, True) so only their
    # descriptions should be recorded -- not the first N by position.
    assert recorded == ["GitHub Trending Repositories, Product Hunt API"]


def test_scrape_all_records_actual_successes_not_positional(monkeypatch):
    """When only the middle scraper succeeds, only its name should be recorded."""
    import scrape_all

    sequence = iter([False, True, False])
    monkeypatch.setattr("scrape_all.run_scraper", lambda name, desc: next(sequence))
    monkeypatch.setattr("scrape_all.init_db", lambda: None)
    recorded = []
    monkeypatch.setattr("scrape_all.record_scrape_completion", lambda summary: recorded.append(summary))

    scrape_all.main()
    assert recorded == ["Hacker News & Show HN"]


def test_scrape_all_main_records_all_successes(monkeypatch):
    import scrape_all

    monkeypatch.setattr("scrape_all.run_scraper", lambda name, desc: True)
    monkeypatch.setattr("scrape_all.init_db", lambda: None)
    recorded = []
    monkeypatch.setattr("scrape_all.record_scrape_completion", lambda summary: recorded.append(summary))

    scrape_all.main()
    assert recorded == ["GitHub Trending Repositories, Hacker News & Show HN, Product Hunt API"]


def test_scrape_all_main_records_no_successes(monkeypatch):
    import scrape_all

    monkeypatch.setattr("scrape_all.run_scraper", lambda name, desc: False)
    monkeypatch.setattr("scrape_all.init_db", lambda: None)
    recorded = []
    monkeypatch.setattr("scrape_all.record_scrape_completion", lambda summary: recorded.append(summary))

    scrape_all.main()
    assert recorded == []


def test_scraper_entrypoints_registry_covers_all_scrapers():
    """SCRAPER_ENTRYPOINTS should map every module used in main()."""
    import scrape_all

    for name in ["scrape_github_trending", "scrape_hackernews", "scrape_producthunt_api"]:
        assert name in scrape_all.SCRAPER_ENTRYPOINTS
    # hackernews has two entry points
    assert len(scrape_all.SCRAPER_ENTRYPOINTS["scrape_hackernews"]) == 2


def test_run_scraper_unknown_module_logs_warning(monkeypatch):
    """An unknown module name should trigger the missing_entrypoint warning."""
    import scrape_all

    factories = {
        "unknown_scraper": lambda module: None,
    }
    configure_loader(monkeypatch, factories)

    assert not scrape_all.run_scraper("unknown_scraper", "Unknown")


def test_scrape_all_main_guard(monkeypatch):
    import importlib.machinery
    import runpy

    factories = {
        "scrape_github_trending": lambda module: module.__dict__.update({"scrape_github_trending": lambda: None}),
        "scrape_hackernews": lambda module: module.__dict__.update(
            {"scrape_hackernews": lambda: None, "scrape_hackernews_show": lambda: None}
        ),
        "scrape_producthunt_api": lambda module: module.__dict__.update({"scrape_producthunt_api": lambda: None}),
    }

    def fake_spec_from_file_location(name, path):
        loader = LoaderWrapper(factories[name])
        return importlib.machinery.ModuleSpec(name, loader)

    def fake_module_from_spec(spec):
        module = spec.loader.create_module(spec)
        module.__spec__ = spec
        return module

    monkeypatch.setattr("importlib.util.spec_from_file_location", fake_spec_from_file_location)
    monkeypatch.setattr("importlib.util.module_from_spec", fake_module_from_spec)
    monkeypatch.setattr("database.init_db", lambda: None)
    monkeypatch.setattr("database.record_scrape_completion", lambda summary: None)

    runpy.run_module("scrape_all", run_name="__main__")
