import sys
from unittest.mock import MagicMock

# Gracefully provide lightweight stubs for optional networking/scraping libraries
# if running in air-gapped test environments where httpx/bs4/playwright are not installed.
for mod_name in ["httpx", "bs4", "playwright", "playwright.sync_api"]:
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()

