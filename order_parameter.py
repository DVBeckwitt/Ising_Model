"""GPU-accelerated 2-D Ising analysis.

This script sweeps through temperatures to compute the magnetisation
and heat capacity of a square Ising lattice.  If CuPy is available a
single ``ElementwiseKernel`` performs each Metropolis sweep on the
GPU, otherwise the code falls back to NumPy on the CPU.

Author: David Beckwitt
"""

from __future__ import annotations

__author__ = "David Beckwitt"

# Progress display (optional)
try:  # try importing tqdm for a nice progress bar
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tqdm = None  # type: ignore

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.optimize import curve_fit

# -------------------------------------------------------------------
# CuPy backend (optional)
# -------------------------------------------------------------------
try:
    import cupy as cp  # type: ignore
    try:
        cp.cuda.runtime.getDeviceCount()
        xp = cp
        GPU = True
        print(">>  CuPy backend selected – running on GPU.")
    except cp.cuda.runtime.CUDARuntimeError:
        xp = np  # type: ignore
        GPU = False
        print(">>  CuPy installed but GPU unavailable – running on CPU (NumPy backend).")
except ModuleNotFoundError:  # CPU fallback
    xp = np  # type: ignore
    GPU = False
    print(">>  CuPy not found – running on CPU (NumPy backend).")

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
J: float = 1.0
KB: float = 1.0

# -------------------------------------------------------------------
# Lattice helpers
# -------------------------------------------------------------------

def init_spins(n: int, ordered: bool = False):
    """Return n×n lattice of ±1 spins."""
    if ordered:
        return xp.ones((n, n), dtype=xp.int8)
    rnd = xp.random.random((n, n))
    return xp.where(rnd < 0.5, 1, -1).astype(xp.int8)

# -------------------------------------------------------------------
# One fused Metropolis sweep (checkerboard)
# -------------------------------------------------------------------
if GPU:
    update_kernel = cp.ElementwiseKernel(
        "raw int8 spins, raw float32 ptab, int32 n, int32 parity, raw float32 rnd",
        "int8 out",
        r"""
        int x = i % n;
        int y = i / n;
        int idx = y * n + x;
        if (((x + y) & 1) != parity) {
            out = spins[idx];
            return;
        }
        int idx_up    = ((y + n - 1) % n) * n + x;
        int idx_down  = ((y + 1) % n) * n + x;
        int idx_left  = y * n + (x + n - 1) % n;
        int idx_right = y * n + (x + 1) % n;
        int s   = spins[idx];
        int nbr = spins[idx_up] + spins[idx_down] + spins[idx_left] + spins[idx_right];
        int dEint = 2 * s * nbr;             // -8,-4,0,4,8
        float p = (dEint <= 0) ? 1.0f : ptab[(dEint + 8) / 4];
        out = (rnd[idx] < p) ? -s : s;
        """,
        "update_kernel",
        reduce_dims=False,
    )

    def sweep(spins: xp.ndarray, ptab: xp.ndarray) -> xp.ndarray:
        n = spins.shape[0]
        rnd = cp.random.random(spins.size, dtype=cp.float32)
        for parity in (0, 1):
            spins = update_kernel(spins.ravel(), ptab, n, parity, rnd, size=spins.size)
        return spins.reshape(n, n)
else:

    def sweep(spins: xp.ndarray, ptab: xp.ndarray) -> xp.ndarray:
        n = spins.shape[0]
        for parity in (0, 1):
            nbr = (
                xp.roll(spins, 1, 0)
                + xp.roll(spins, -1, 0)
                + xp.roll(spins, 1, 1)
                + xp.roll(spins, -1, 1)
            )
            dE = 2 * spins * nbr
            rnd = xp.random.random(spins.shape)
            accept = (dE <= 0) | (rnd < xp.exp(-ptab[(dE + 8) // 4]))
            mask = ((xp.indices(spins.shape).sum(0) & 1) == parity)
            spins = xp.where(mask & accept, -spins, spins)
        return spins

# -------------------------------------------------------------------
# Observables
# -------------------------------------------------------------------

def mag(spins: xp.ndarray) -> xp.ndarray:
    return xp.abs(spins.sum()) / spins.size


def energy_per_spin(spins: xp.ndarray) -> xp.ndarray:
    nbr = (
        xp.roll(spins, 1, 0)
        + xp.roll(spins, -1, 0)
        + xp.roll(spins, 1, 1)
        + xp.roll(spins, -1, 1)
    )
    E = -J * xp.sum(spins * nbr) / 2.0
    return E / spins.size

# -------------------------------------------------------------------
# Logistic magnetisation fit function
# -------------------------------------------------------------------

def M_fit(T: xp.ndarray, Tc: float, a: float) -> xp.ndarray:
    """Return 1/2 (1 + tanh(a * (Tc - T)))."""
    T = xp.asarray(T, dtype=float)
    return 0.5 * (1.0 + xp.tanh(a * (Tc - T)))


# -------------------------------------------------------------------
# Asymmetric magnetisation fit (piece-wise tanh)
# -------------------------------------------------------------------
def M_fit_asym(
    T: xp.ndarray,
    Tc: float,
    a_minus: float,
    a_plus: float,
    T_split: float | None = None,
    linear_width: float = 0.0,
) -> xp.ndarray:
    r"""Return piece-wise logistic magnetisation with optional linear splice.

    Parameters
    ----------
    T : array-like
        Temperatures at which to evaluate the fit function.
    Tc : float
        Transition temperature controlling the steepness of the sigmoids.
    a_minus : float
        Slope parameter for :math:`T < T_{split}`.
    a_plus : float
        Slope parameter for :math:`T \ge T_{split}`.
    T_split : float, optional
        Temperature at which the piece-wise splice occurs.  If ``None``
        (default), ``Tc`` is used.
    linear_width : float, optional
        Width of the linear section connecting the left and right sigmoids.
        If ``0`` (default) the splice is sharp.
    """

    T = xp.asarray(T, dtype=float)
    left = 0.5 * (1.0 + xp.tanh(a_minus * (Tc - T)))
    right = 0.5 * (1.0 + xp.tanh(a_plus * (Tc - T)))
    if T_split is None:
        T_split = Tc

    if linear_width <= 0.0:
        return xp.where(T < T_split, left, right)

    half = 0.5 * linear_width
    left_mask = T < (T_split - half)
    right_mask = T > (T_split + half)

    out = xp.where(left_mask, left, xp.where(right_mask, right, 0.0))

    left_val = float(0.5 * (1.0 + xp.tanh(a_minus * (Tc - (T_split - half)))))
    right_val = float(0.5 * (1.0 + xp.tanh(a_plus * (Tc - (T_split + half)))))
    slope = (right_val - left_val) / linear_width
    mid_mask = ~(left_mask | right_mask)
    out = xp.where(mid_mask, left_val + slope * (T - (T_split - half)), out)
    return out


def to_np(arr):
    """Return a NumPy array regardless of backend."""
    if GPU:
        return cp.asnumpy(arr)
    return np.asarray(arr)


# -------------------------------------------------------------------
# Thermodynamic sweep
# -------------------------------------------------------------------

def compute_thermodynamic_curves(
    size: int = 256,
    temperatures: np.ndarray = np.linspace(0.5, 5.0, 100),
    equil_steps: int = 1000,
    sample_steps: int = 2000,
    show_progress: bool = True,
):
    temps = np.asarray(temperatures, dtype=float)
    spins = init_spins(size, ordered=True)
    mags = []
    cvs = []

    iterator = temps
    if show_progress:
        if tqdm is not None:
            iterator = tqdm(temps, desc="Temperature sweep")
        else:
            print("-- Temperature sweep --")

    for idx, T in enumerate(iterator):
        if show_progress and tqdm is None:
            step = idx + 1
            total = len(temps)
            pct = 100 * step / total
            print(f"  {step}/{total}  (T={T:.2f})  {pct:5.1f}%", end="\r")
        beta = 1.0 / (KB * T)
        # probability lookup for ΔE=4 and ΔE=8 (indices 3 and 4)
        ptab = xp.ones(5, dtype=xp.float32)
        ptab[3] = xp.exp(-beta * 4.0)
        ptab[4] = xp.exp(-beta * 8.0)

        for _ in range(equil_steps):
            spins = sweep(spins, ptab)

        m_samples = []
        e_samples = []
        for _ in range(sample_steps):
            spins = sweep(spins, ptab)
            m_samples.append(mag(spins))
            e_samples.append(energy_per_spin(spins))

        m_samples = xp.asarray(m_samples)
        e_samples = xp.asarray(e_samples)

        mags.append(m_samples.mean())
        mean_e = e_samples.mean()
        var_e = (e_samples ** 2).mean() - mean_e ** 2
        cvs.append(var_e / (KB * T ** 2))
    if show_progress and tqdm is None:
        print()

    return temps, to_np(xp.array(mags)), to_np(xp.array(cvs))


# -------------------------------------------------------------------
# Plotting and fitting
# -------------------------------------------------------------------

def main():
    temps, mags, cvs = compute_thermodynamic_curves(show_progress=True)

    fig, (ax_mag, ax_cv) = plt.subplots(2, 1, sharex=True, dpi=150)
    plt.subplots_adjust(bottom=0.25)
    ax_mag.scatter(temps, mags, s=25, color="blue", label="data")

    mask = mags > 0.05
    try:
        popt, _ = curve_fit(
            lambda T, Tc, a_m, a_p, T_split, width: to_np(
                M_fit_asym(T, Tc, a_m, a_p, T_split, width)
            ),
            temps[mask],
            mags[mask],
            p0=(2.27, 1.0, 0.7, 2.27, 0.0),
            maxfev=10000,
        )
    except Exception as exc:
        print(f"Fit failed: {exc}")
        popt = (2.27, 1.0, 0.7, 2.27, 0.0)

    Tc_fit, a_minus_fit, a_plus_fit, split_fit, width_fit = popt
    T_fit = np.linspace(temps.min(), temps.max(), 200)
    line_fit, = ax_mag.plot(
        T_fit,
        to_np(M_fit_asym(T_fit, Tc_fit, a_minus_fit, a_plus_fit, split_fit,
                        width_fit)),
        color="red",
        label="fit",
    )

    param_box = ax_mag.text(
        0.05,
        0.95,
        "",
        transform=ax_mag.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax_tc = plt.axes([0.25, 0.20, 0.65, 0.03])
    slider_tc = Slider(ax_tc, "Tc", temps.min(), temps.max(), valinit=Tc_fit)

    ax_a_minus = plt.axes([0.25, 0.15, 0.65, 0.03])
    slider_a_m = Slider(ax_a_minus, "a₋ (ordered)", 0.1, 20.0,
                        valinit=a_minus_fit)

    ax_a_plus = plt.axes([0.25, 0.10, 0.65, 0.03])
    slider_a_p = Slider(ax_a_plus, "a₊ (disordered)", 0.1, 2000.0,
                        valinit=a_plus_fit)

    ax_split = plt.axes([0.25, 0.05, 0.65, 0.03])
    slider_split = Slider(
        ax_split,
        "T_split",
        temps.min(),
        temps.max(),
        valinit=split_fit,
    )

    ax_width = plt.axes([0.25, 0.00, 0.65, 0.03])
    slider_width = Slider(
        ax_width,
        "width",
        0.0,
        1.0,
        valinit=width_fit,
        valstep=0.01,
    )

    def update_text():
        param_box.set_text(
            f"Tc = {slider_tc.val:.3f} J/$k_B$\n"
            f"a₋ = {slider_a_m.val:.3f}\n"
            f"a₊ = {slider_a_p.val:.3f}\n"
            f"T_split = {slider_split.val:.3f}\n"
            f"width = {slider_width.val:.2f}"
        )

    def update_curve(val=None):
        line_fit.set_ydata(
            to_np(
                M_fit_asym(
                    T_fit,
                    slider_tc.val,
                    slider_a_m.val,
                    slider_a_p.val,
                    slider_split.val,
                    slider_width.val,
                )
            )
        )
        update_text()
        fig.canvas.draw_idle()

    update_curve()
    slider_tc.on_changed(update_curve)
    slider_a_m.on_changed(update_curve)
    slider_a_p.on_changed(update_curve)
    slider_split.on_changed(update_curve)
    slider_width.on_changed(update_curve)

    ax_btn = plt.axes([0.025, 0.1, 0.15, 0.06])
    btn = Button(ax_btn, "Refine")

    def refine(event):
        p0 = (
            slider_tc.val,
            slider_a_m.val,
            slider_a_p.val,
            slider_split.val,
            slider_width.val,
        )
        try:
            new_popt, _ = curve_fit(
                lambda T, Tc, a_m, a_p, T_split, width: to_np(
                    M_fit_asym(T, Tc, a_m, a_p, T_split, width)
                ),
                temps[mask],
                mags[mask],
                p0=p0,
                maxfev=10000,
            )
            slider_tc.set_val(new_popt[0])
            slider_a_m.set_val(new_popt[1])
            slider_a_p.set_val(new_popt[2])
            slider_split.set_val(new_popt[3])
            slider_width.set_val(new_popt[4])
        except Exception as exc:
            print(f"Refit failed: {exc}")

    btn.on_clicked(refine)

    ax_mag.set_ylabel("|M|")
    ax_mag.set_title("|M| vs Temp")
    ax_mag.legend()
    ax_mag.grid(True)

    ax_cv.plot(temps, cvs, color="green")
    ax_cv.set_xlabel("Temp (J/$k_B$)")
    ax_cv.set_ylabel("Heat capacity / spin")
    ax_cv.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
