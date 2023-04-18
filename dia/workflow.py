import logging
from dataclasses import dataclass

from dia.config import Config
from dia.peaker import PeakPicker

_logger = logging.getLogger(__package__)
_info = _logger.info


@dataclass
class DiaImsWorkflow:
    config: Config
    """The config."""

    def process(self, f: str):
        _info("peak-picking...")
        PeakPicker(self.config).pick_peaks(f)
