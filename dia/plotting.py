import typing

import numpy as np
import pyopenms as ms
from matplotlib import pyplot as plt


# noinspection SpellCheckingInspection
def set_global_style(font="Source Han Serif", math_font="dejavuserif", font_size=16, fig_size=(10, 6)):
    plt.rcParams["font.family"] = font
    plt.rcParams["mathtext.fontset"] = math_font
    plt.rcParams["font.size"] = font_size
    plt.rcParams["figure.figsize"] = fig_size


def subplots() -> tuple[plt.Figure, plt.Axes]:
    return plt.subplots()


def plot_peaks(ax: plt.Axes, x: np.ndarray, y: np.ndarray, width: float, **kwargs):
    ax.plot(
        np.stack((x - width/2, x, x + width/2)).T.reshape(-1),
        np.stack((np.zeros_like(y), y, np.zeros_like(y))).T.reshape(-1),
        **kwargs,
    )


def scatter_map(exp: ms.MSExperiment):
    fig, ax = subplots()
    mzs, dts, intensities = np.array(([], [], []))
    peaks: ms.MSSpectrum
    for peaks in exp:
        mz, intensity = peaks.get_peaks()
        mzs = np.concatenate((mzs, mz))
        intensities = np.concatenate((intensities, intensity))
        dts = np.concatenate((dts, np.ones_like(mz) * peaks.getDriftTime()))
    fig.colorbar(ax.scatter(mzs, dts, s=intensities/intensities.max()*5, c=intensities))
    ax.set_xlabel("$m/z$")
    ax.set_ylabel("Relative Drift Time")
    fig.show()
    plt.show()


def show_raw_spectrum(orig: typing.Optional[ms.MSSpectrum] = None, peaks: typing.Optional[ms.MSSpectrum] = None):
    fig, ax = subplots()
    fig.tight_layout()
    ax.set_xlabel("$m/z$")
    ax.set_ylabel("Intensity")

    if orig is not None:
        mz, intensity = orig.get_peaks()
        ax.plot(mz, intensity, "black", linewidth=0.5)

    if peaks is not None:
        mz, intensity = peaks.get_peaks()
        if orig is None:
            plot_peaks(ax, mz, intensity, 0.02)
            ax.set_title(peaks.getName())
        else:
            ax.plot(mz, intensity, "ro", ms=2.0)

    fig.show()
    plt.show()
