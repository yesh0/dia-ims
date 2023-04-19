import logging
from dataclasses import dataclass

import pyopenms as ms
import tqdm

from dia import plotting
from dia.config import Config
from dia.featurer import FeatureFinder, FeatureIntensityMap
from dia.peaker import PeakPicker

_logger = logging.getLogger(__package__)
_info = _logger.info


@dataclass
class DiaImsWorkflow:
    config: Config
    """The config."""

    def process(self, f: str):
        _info("processing %s", f)
        _info("peak-picking...")
        peak1_map, peak2_map = PeakPicker(self.config).pick_peaks(f)
        plotting.scatter_map(peak1_map)
        plotting.scatter_map(peak2_map)
        ff = FeatureFinder(self.config)
        _info("feature finding...")
        feature_maps = dict([
            (i + 1, FeatureIntensityMap(ff.find(peak_map, f, i + 1), peak_map))
            for i, peak_map in enumerate(tqdm.tqdm((peak1_map, peak2_map)))
        ])
        self.plot_features(feature_maps)
        self.plot_feature_heatmap(feature_maps)

    def plot_features(self, feature_maps: dict[int, FeatureIntensityMap]):
        if self.config.require(int, "feature_finder", "debug") >= 2:
            for feature_map in feature_maps.values():
                feature: ms.Feature
                for feature in feature_map.feature_map:
                    if feature.getCharge() >= 2:
                        feature_map.plot_feature(feature)

    def plot_feature_heatmap(self, feature_maps: dict[int, FeatureIntensityMap]):
        if self.config.require(int, "feature_finder", "debug") >= 1:
            for feature_map in feature_maps.values():
                feature_map.plot_feature_heatmap()
