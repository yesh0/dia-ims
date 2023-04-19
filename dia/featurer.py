import bisect
import os.path
from dataclasses import dataclass

import numpy as np
import pyopenms as ms
from matplotlib import pyplot as plt
from scipy.spatial import KDTree

from dia import plotting
from dia.config import Config


@dataclass
class FeatureFinder:
    config: Config
    """The config."""

    def find(self, peak_map: ms.MSExperiment, f: str, level: int):
        feature_map = ms.FeatureMap()

        caches = self.config.require(bool, "feature_finder", "cache")
        cache = self._get_cache_file(f, level)
        if caches and os.path.exists(cache):
            if os.stat(f).st_mtime < os.stat(cache).st_mtime:
                reader = ms.FeatureXMLFile()
                reader.load(cache, feature_map)
                return feature_map

        ff = ms.FeatureFinder()
        seeds = ms.FeatureMap()
        params = ms.FeatureFinderAlgorithmPicked().getParameters()
        params.setValue("feature:min_score", self.config.require(float, "feature_finder", "min_score"))
        configured_params = (
            (float, "mass_trace", "mz_tolerance"),
            (int, "mass_trace", "min_spectra"),
            (float, "mass_trace", "slope_bound"),
            (int, "isotopic_pattern", "charge_low"),
            (int, "isotopic_pattern", "charge_high"),
            (float, "isotopic_pattern", "mz_tolerance"),
            (float, "isotopic_pattern", "intensity_percentage"),
        )
        for t, category, item in configured_params:
            params.setValue(
                f"{category}:{item}",
                self.config.require(t, "feature_finder", category, item),
            )
        # RuntimeError: FeatureFinder can only operate on MS level 1 data. Please do not use MS/MS data. Aborting.
        # Just to avoid the above error.
        self._set_ms_level(peak_map, 1)
        # noinspection SpellCheckingInspection
        ff.run(
            "centroided", peak_map, feature_map,
            params, seeds,
        )
        feature_map.updateRanges()
        feature_map.sortByOverallQuality()
        feature_map.setUniqueIds()
        writer = ms.FeatureXMLFile()
        writer.store(cache, feature_map)
        return feature_map

    @classmethod
    def _get_cache_file(cls, f: str, level: int):
        original = os.path.realpath(f)
        dir_path = os.path.dirname(original)
        name = os.path.basename(f).split(".")[0]
        return os.path.join(dir_path, f"{name}.MS{level}.featureXML")

    @classmethod
    def _set_ms_level(cls, exp: ms.MSExperiment, level: int):
        spectra = []
        s: ms.MSSpectrum
        for s in exp:
            s.setMSLevel(level)
            spectra.append(s)
        exp.clear(False)
        exp.clearRanges()
        for s in spectra:
            exp.addSpectrum(s)
        exp.updateRanges()


class FeatureIntensityMap:
    """A feature map with intensity info, which FeatureMap in OpenMS lacks."""

    def __init__(self, feature_map: ms.FeatureMap, peak_map: ms.MSExperiment):
        self.mzs, self.rts, self.intensities = np.array([[], [], []])
        peaks: ms.MSSpectrum
        for peaks in peak_map:
            mz, intensity = peaks.get_peaks()
            rt = peaks.getRT() * np.ones_like(mz)
            self.mzs = np.concatenate((self.mzs, mz))
            self.intensities = np.concatenate((self.intensities, intensity))
            self.rts = np.concatenate((self.rts, rt))
        self.tree = KDTree(np.array([self.rts, self.mzs]).T)
        self.feature_map = feature_map
        self.peak_map = peak_map

    def __getitem__(self, key: tuple[float, float]):
        rt, mz = key
        return self.intensities[self.tree.query((rt, mz))[1]]

    def query_peaks(self, rt: float, region: tuple[float, float]):
        s: ms.MSSpectrum = [s for s in self.peak_map if abs(s.getRT() - rt) < 0.001][0]
        mz, intensity = s.get_peaks()
        left, right = bisect.bisect_left(mz, region[0]), bisect.bisect_left(mz, region[1])
        return mz[left:right], intensity[left:right]

    def plot_feature(self, feature: ms.Feature):
        hulls = []
        for hull in feature.getConvexHulls():
            points = np.array(hull.getHullPoints())
            intensity = np.array([self[p] for p in points])
            hulls.append((points, intensity))
        i = np.argmax(intensity.sum() for _, intensity in hulls)
        j = np.argmax(hulls[i][1])
        max_rt = hulls[i][0][j][0]
        peaks_at_max_rt = []
        for points, intensity in hulls:
            indices = np.argwhere(points.T[0] == max_rt)
            if len(indices) > 0:
                i = indices.reshape(-1)[0]
                peaks_at_max_rt.append((points[i, 1], intensity[i]))

        peaks = np.array(peaks_at_max_rt).T
        fig, ax = plotting.subplots()
        fig.tight_layout()
        original = self.query_peaks(max_rt, (peaks[0][0] - 1, peaks[0][-1] + 1))
        plotting.plot_peaks(ax, original[0], original[1], 0.02)
        plotting.plot_peaks(ax, peaks[0], peaks[1], 0.02, "fit")
        ax.legend()
        fig.show()
        plt.show()

    def plot_feature_heatmap(self):
        fig, ax = plotting.subplots()
        mzs, rts, intensities = np.array([[], [], []])
        for peaks in self.peak_map:
            mz, intensity = peaks.get_peaks()
            mzs = np.concatenate((mzs, mz))
            rts = np.concatenate((rts, peaks.getRT() * np.ones_like(mz)))
            intensities = np.concatenate((intensities, intensity))
        heap_map, _, _ = np.histogram2d(mzs, rts, bins=[400, int(self.peak_map.size() / 3)],
                                        weights=np.log10(intensities))
        fig.tight_layout()
        t = ax.imshow(heap_map.T, cmap="gnuplot2", aspect="auto", origin="lower")
        ax.set_xlabel("$m/z$")
        ax.set_ylabel("Relative RT")
        fig.colorbar(t).ax.set_title("lg $I$")
        fig.show()
        plt.show()
