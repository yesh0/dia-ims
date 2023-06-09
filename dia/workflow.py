import logging
from dataclasses import dataclass

import pyopenms as ms
import tqdm

from dia import plotting
from dia.config import Config
from dia.featurer import FeatureFinder, FeatureIntensityMap
from dia.matcher import TandemMatcher
from dia.peaker import PeakPicker
from dia.searcher import ProteinLibrarySearcher

_logger = logging.getLogger(__package__)
_info = _logger.info


@dataclass
class DiaImsWorkflow:
    config: Config
    """The config."""

    def process(self, f: str):
        should_plot = self.config.require(bool, "plot")

        _info("processing %s", f)
        _info("peak-picking...")
        peak1_map, peak2_map = PeakPicker(self.config).pick_peaks(f)

        if should_plot:
            plotting.scatter_map(peak1_map)
            plotting.scatter_map(peak2_map)

        ff = FeatureFinder(self.config)
        _info("feature finding...")
        feature_maps = dict([
            (i + 1, FeatureIntensityMap(ff.find(peak_map, f, i + 1), peak_map))
            for i, peak_map in enumerate(tqdm.tqdm((peak1_map, peak2_map)))
        ])

        if should_plot:
            self.plot_features(feature_maps)
            self.plot_real_feature_heatmap(feature_maps)
            self.plot_feature_heatmap(feature_maps)

        _info("peptide searching...")
        searcher = ProteinLibrarySearcher(self.config)
        peptides = searcher.search(f, feature_maps[1])

        _info("performing deconvolution...")
        matcher = TandemMatcher(self.config)
        matches = matcher.match(feature_maps[1], feature_maps[2])
        deconvoluted, proteins = matcher.organize(feature_maps[1], feature_maps[2], matches, peptides)
        matcher.stat(f, feature_maps[1], feature_maps[2], deconvoluted, proteins)

    def plot_features(self, feature_maps: dict[int, FeatureIntensityMap]):
        if self.config.require(int, "feature_finder", "debug") >= 2:
            for feature_map in feature_maps.values():
                feature: ms.Feature
                for feature in feature_map.feature_map:
                    if feature.getCharge() >= 2:
                        feature_map.plot_feature(feature)

    def plot_real_feature_heatmap(self, feature_maps: dict[int, FeatureIntensityMap]):
        if self.config.require(int, "feature_finder", "debug") >= 1:
            import numpy as np
            mzs, dts, intensities = np.array(([], [], []))
            for feature_map in feature_maps.values():
                feature: ms.Feature
                for feature in feature_map.feature_map:
                    dt, mz = np.array(feature.getConvexHull().getHullPoints()).T
                    intensity = []
                    for d, m in zip(dt, mz):
                        intensity.append(feature_map[d, m])
                    mzs = np.concatenate((mzs, mz))
                    dts = np.concatenate((dts, dt))
                    intensities = np.concatenate((intensities, intensity))
                from matplotlib import pyplot as plt
                fig, ax = plt.subplots()
                fig.colorbar(ax.scatter(mzs, dts, s=intensities / intensities.max() * 5, c=intensities))
                ax.set_xlabel("$m/z$")
                ax.set_ylabel("Relative Drift Time")
                fig.show()
                plt.show()

    def plot_feature_heatmap(self, feature_maps: dict[int, FeatureIntensityMap]):
        if self.config.require(int, "feature_finder", "debug") >= 1:
            for feature_map in feature_maps.values():
                feature_map.plot_feature_heatmap()
