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


def show_raw_spectrum(orig: ms.MSSpectrum, peaks: ms.MSSpectrum):
    fig, ax = subplots()
    fig.tight_layout()
    ax.set_xlabel("$m/z$")
    ax.set_ylabel("Intensity")

    mz, intensity = orig.get_peaks()
    ax.plot(mz, intensity, "black", linewidth=0.5)

    mz, intensity = peaks.get_peaks()
    ax.plot(mz, intensity, "ro", ms=2.0)

    fig.show()
    plt.show()
