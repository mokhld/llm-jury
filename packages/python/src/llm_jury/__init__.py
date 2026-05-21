from ._version import __version__
from .calibration.optimizer import ThresholdCalibrator
from .debate.engine import DebateConfig, DebateMode, DebateTranscript
from .judges.base import Verdict
from .jury.core import Jury, JuryStats
from .personas.base import Persona, PersonaResponse
from .personas.registry import PersonaRegistry

__all__ = [
    "__version__",
    "DebateConfig",
    "DebateMode",
    "DebateTranscript",
    "Jury",
    "JuryStats",
    "Persona",
    "PersonaResponse",
    "PersonaRegistry",
    "ThresholdCalibrator",
    "Verdict",
]
