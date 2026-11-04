import importlib
import pkgutil
from typing import Dict, List, Type
from app.core.contracts import BaseAnalyzer

class AnalyzerRegistry:
    _plugins: Dict[str, BaseAnalyzer] = {}

    @classmethod
    def register(cls, tier: int):
        """Decorator that registers an analyzer class into a specific execution tier."""
        def wrapper(analyzer_cls: Type[BaseAnalyzer]):
            analyzer_cls.tier = tier
            instance = analyzer_cls()
            cls._plugins[instance.name] = instance
            return analyzer_cls
        return wrapper

    @classmethod
    def get_ordered_pipeline(cls) -> List[BaseAnalyzer]:
        """Returns all registered analyzers sorted by tier (Tier 1 -> Tier 2 -> Tier 3 -> Tier 4)."""
        return sorted(cls._plugins.values(), key=lambda a: a.tier)

    @classmethod
    def get_by_name(cls, name: str) -> BaseAnalyzer:
        return cls._plugins[name]

    @classmethod
    def autodiscover(cls, package_name: str = "app.analyzers") -> None:
        """Automatically imports all modules in app/analyzers/ so decorators execute."""
        package = importlib.import_module(package_name)
        for _, module_name, _ in pkgutil.iter_modules(package.__path__):
            if module_name != "base":
                importlib.import_module(f"{package_name}.{module_name}")