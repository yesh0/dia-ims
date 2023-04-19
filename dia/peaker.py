import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pyopenms as ms
import tqdm
from scipy.signal import find_peaks, savgol_filter, argrelmin

from dia import plotting, utils
from dia.config import Config

_logger = logging.getLogger(__package__)
_info = _logger.info


@dataclass
class PeakPicker:
    config: Config
    """The config."""

    @classmethod
    def classify_peaks(cls, exp: ms.OnDiscMSExperiment):
        """
        Groups spectra by their drift time.

        In our IMS experiment, each IMS cycle produces N spectra,
        and we repeat for M times.

        Converted mzML files (from Waters RAW by ProteoWizard) seems to have
        M distinct retention time and N distinct drift time. While most applications
        merge every N spectra into one, resulting in M spectra in retention time,
        we actually want merging every M spectra to get data with respect to drift time.
        So we have to do the merging ourselves, instead of using MsConvert to do so.
        """
        spectrum_count = exp.getMetaData().size()
        drift_times = {}
        for i in tqdm.trange(spectrum_count):
            s = exp.getSpectrum(i)
            dt = s.getDriftTime()
            if dt not in drift_times:
                drift_times[dt] = ([], [])
            drift_times[dt][s.getMSLevel() - 1].append(i)
        _info("drift time counts: %d", len(drift_times))
        return drift_times

    def pick_peaks(self, f: str) -> tuple[ms.MSExperiment, ms.MSExperiment]:
        peak1_map, peak2_map = ms.MSExperiment(), ms.MSExperiment()
        exporter = ms.MzMLFile()

        caches = self.config.require(bool, "peaker", "cache")
        debug = self.config.require(int, "peaker", "debug") >= 1
        picker = self.config.require(str, "peaker", "picker")
        tic_threshold = self.config.require(float, "peaker", "tic_threshold")
        if picker not in ("hires", "simple"):
            raise ValueError("wrong configuration peaker.picker, expecting \"hires\" or \"simple\"")

        # Cache or not.
        ms1_cache, ms2_cache = utils.get_cache_files(f, "peaks1.mzML", "peaks2.mzML")
        if caches and utils.is_cache_recent(f, ms1_cache, ms2_cache):
            exporter.load(ms1_cache, peak1_map)
            exporter.load(ms2_cache, peak2_map)
            return peak1_map, peak2_map

        exp = ms.OnDiscMSExperiment()
        if not exp.openFile(f):
            raise ValueError("possibly wrong file format, expecting an mzML file")

        bins = self.classify_peaks(exp)
        progress = tqdm.tqdm(bins.items())
        for bin_i, (ms1_spectra, ms2_spectra) in progress:
            raw1 = [exp.getSpectrum(i) for i in ms1_spectra]
            raw2 = [exp.getSpectrum(i) for i in ms2_spectra]

            merged1 = self.merged_spectra(raw1)
            merged2 = self.merged_spectra(raw2)

            for raw, exp in zip((merged1, merged2), (peak1_map, peak2_map)):
                progress.set_description(f"MS{raw.getMSLevel()}")
                total_ion_count = raw.calculateTIC()
                if total_ion_count < tic_threshold:
                    continue
                if picker == "hires":
                    s = self.hires_peaking(raw)
                else:
                    s = self.simplistic_peaking(raw)
                if debug:
                    plotting.show_raw_spectrum(raw, s)
                exp.addSpectrum(s)
        peak1_map.updateRanges()
        peak2_map.updateRanges()

        exporter.store(ms1_cache, peak1_map)
        exporter.store(ms2_cache, peak2_map)

        return peak1_map, peak2_map

    def simplistic_peaking(self, s: ms.MSSpectrum):
        """
        Picks the peaks from the spectrum (of non-uniform data) following a really simplistic algorithm.

        0. We assume that
           1) the data is non-uniform and sparse such that consecutive zero data points are removed,
           2) and the data is relatively uniform where data points are dense.
        1. The algorithm groups data points by their m/z distance:
           one can always find another data point near a data point in a group within `distance`.
        2. If a group is within `max_peak_width`, they are considered to contain only one peak.
        3. If a group spans more than `max_peak_width`, they then get smoothed and peak-detected
           to separate multiple peaks from the group (with `savgol_filter` and `find_peaks`).
           If the peak detection fails to find any, default to `2.`.
        """
        distance = self.config.require(float, "peaker", "simple", "distance")
        threshold = self.config.require(float, "peaker", "simple", "threshold")
        max_peak_width = self.config.require(float, "peaker", "simple", "max_peak_width")
        debug = self.config.require(int, "peaker", "debug") >= 2
        mz, intensity = s.get_peaks()
        i, j = 0, 0
        peak_mz, peak_i = [], []
        while i < len(mz):
            while j < len(mz) and mz[j] - mz[max(j - 1, i)] < distance:
                j += 1
            # Not applicable since the grid is not uniform:
            # area = (mz[i + 1:j] - mz[i:j - 1]).dot(intensity[i:j - 1] + intensity[i + 1:j]) / 2
            area = intensity[i:j].sum()
            if area > threshold:
                if mz[j - 1] - mz[i] < max_peak_width:
                    peak_mz.append(np.average(mz[i:j], weights=intensity[i:j]))
                    peak_i.append(area)
                else:
                    smoothed = intensity[i:j]
                    for _ in range(3):
                        smoothed = savgol_filter(smoothed, 6, 2, mode="constant")
                    indices, _ = find_peaks(
                        smoothed,
                        distance=max(max_peak_width / 2 / (mz[1:] - mz[:-1]).mean(), 3),
                        height=threshold,
                        prominence=3,
                    )
                    indices = np.array(indices)
                    if debug and len(indices) != 0:
                        fig, ax = plotting.subplots()
                        ax.clear()
                        ax.plot(mz[i:j], intensity[i:j], label="raw")
                        ax.plot(mz[i:j], smoothed, label="smoothed")
                        ax.legend(frameon=False)
                        ax.scatter(mz[i + indices], smoothed[indices])
                    else:
                        fig, ax = None, None
                    for index in indices:
                        peak_mz.append(mz[i + index])
                        minimals = argrelmin(smoothed[index::-1])[0]
                        left = (index - minimals[0]) if len(minimals) > 0 else 0
                        minimals = argrelmin(smoothed[index:])[0]
                        right = (index + minimals[0]) if len(minimals) > 0 else len(smoothed) - 1
                        peak_i.append(intensity[i + left:i + right + 1].sum())
                        if debug:
                            ax.scatter([mz[i + left], mz[i + right]], [smoothed[left], smoothed[right]])
                    if debug and len(indices) != 0:
                        fig.show()
                        fig.show()
                        plt.close(fig)
                    if len(indices) == 0:
                        peak_mz.append(np.average(mz[i:j], weights=intensity[i:j]))
                        peak_i.append(area)
            i = j
        peaks = ms.MSSpectrum(s)
        peaks.clear(False)
        peaks.clearRanges()
        peaks.set_peaks((peak_mz, peak_i))
        peaks.updateRanges()
        return peaks

    def hires_peaking(self, s: ms.MSSpectrum):
        picker = ms.PeakPickerHiRes()
        params = picker.getParameters()
        params.setValue("signal_to_noise", self.config.require(float, "peaker", "hires", "signal_to_noise"))
        picker.setParameters(params)
        peaks = ms.MSSpectrum(s)
        peaks.clear(False)
        peaks.clearRanges()
        picker.pick(s, peaks)
        peaks.updateRanges()
        return peaks

    @classmethod
    def sort_spectrum(cls, mz: np.ndarray, intensity: np.ndarray):
        indices = np.argsort(mz)
        mz, intensity = mz[indices], intensity[indices]
        merged_mz, merged_intensity = [], []
        for mz_value, i in zip(mz, intensity):
            if len(merged_mz) == 0 or merged_mz[-1] != mz_value:
                merged_mz.append(mz_value)
                merged_intensity.append(i)
            else:
                merged_intensity[-1] += i
        return merged_mz, merged_intensity

    def merged_spectra(self, merging_spectra: list[ms.MSSpectrum]):
        merged = ms.MSSpectrum(merging_spectra[0])
        merged.clear(False)
        merged.clearRanges()
        mzs, intensities = zip(*(s.get_peaks() for s in merging_spectra))
        merged.set_peaks(self.sort_spectrum(np.concatenate(mzs), np.concatenate(intensities)))
        merged.setRT(merged.getDriftTime())
        merged.updateRanges()
        return merged
