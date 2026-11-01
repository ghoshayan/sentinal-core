from abc import ABC, abstractmethod
from typing import Any
from app.schemas import AnalysisResult

class BaseAnalyzer(ABC):
    """Strict interface contract for all classification modules."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def analyze(self, payload: Any) -> AnalysisResult:
        """Process the input payload and return a standardized AnalysisResult."""
        pass