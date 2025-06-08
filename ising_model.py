"""Interactive 2D Ising model viewer.

This module simulates a square Ising lattice using a Metropolis
algorithm accelerated with Numba.  The animation provides controls for
temperature, lattice size and speed.  Energy is tracked in real time.

Author: David Beckwitt
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider, Button
from numba import njit, prange
from typing import Any, List

# Physical constants (k_B = 1, J = 1)
KB = 1.0
J = 1.0
# Exact critical temperature for the infinite 2D square lattice
# (finite lattices will be slightly lower)
TC_INF = 2 * J / np.log(1 + np.sqrt(2))
@njit(cache=True, fastmath=True, parallel=True)
def energy_per_spin(spins: np.ndarray, J: float = 1.0) -> float:
    """Return the average energy per spin."""
    return compute_energy(spins, J) / spins.size


def critical_temperature(size: int, J: float = 1.0) -> float:
    """Return the finite-size critical temperature."""
    return TC_INF * (1 - 1.0 / size)



def initialize_spins(size: int = 50, ordered: bool = False) -> np.ndarray:
    """Initialize a square lattice.

    Parameters
    ----------
    size : int
        Linear dimension of the lattice.
    ordered : bool, optional
        If True, start from an ordered state with all spins up. Otherwise
        initialize spins randomly. The default is False.
    """

    if ordered:
        return np.ones((size, size), dtype=np.int8)
    return np.random.choice([-1, 1], size=(size, size))


@njit(parallel=True, cache=True, fastmath=True)
def compute_energy(spins: np.ndarray, J: float = 1.0) -> float:
    """Compute the Ising energy with nearest-neighbor interactions."""
    size = spins.shape[0]
    energy = 0.0
    # Periodic boundary conditions
    for i in prange(size):
        for j in range(size):
            s = spins[i, j]
            # interact with right and down neighbors to avoid double counting
            right = spins[i, (j + 1) % size]
            down = spins[(i + 1) % size, j]
            energy -= J * s * (right + down)
    return energy


@njit(parallel=True, cache=True, fastmath=True)
def metropolis_step(spins: np.ndarray, beta: float = 1.0, J: float = 1.0) -> np.ndarray:
    """Perform a single Metropolis sweep with checkerboard updates."""
    size = spins.shape[0]
    for parity in range(2):
        for i in prange(size):
            for j in range(size):
                if (i + j) % 2 == parity:
                    s = spins[i, j]
                    neighbors = (
                        spins[(i + 1) % size, j]
                        + spins[(i - 1) % size, j]
                        + spins[i, (j + 1) % size]
                        + spins[i, (j - 1) % size]
                    )
                    delta_e = 2 * J * s * neighbors
                    if delta_e <= 0 or np.random.rand() < np.exp(-beta * delta_e):
                        spins[i, j] = -s
    return spins


def main() -> None:
    """Launch the interactive Ising model viewer."""

    size = 50
    temp_init = 1.0
    steps = 200

    spins = initialize_spins(size)

    # Compile numba functions ahead of time to avoid delays during animation
    _ = compute_energy(spins)
    _ = metropolis_step(spins.copy(), 1.0 / (KB * temp_init), J)

    fig, (ax, ax_energy) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(left=0.25, bottom=0.25)

    im = ax.imshow(spins, cmap="binary", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title("2D Ising Model")
    ax.axis("off")

    energies = []
    times = []

    (line_energy,) = ax_energy.plot([], [], color="blue")
    ax_energy.set_title("Energy vs Time")
    ax_energy.set_xlabel("Time step")
    ax_energy.set_ylabel("Energy")

    def update(frame: int) -> List[Any]:
        beta = 1.0 / (KB * temp_slider.val)
        for _ in range(2):        # doubles decorrelation, minor cost
            metropolis_step(spins, beta, J)        
            
        im.set_data(spins)

        energy = compute_energy(spins, J)
        energies.append(energy)
        times.append(frame)
        line_energy.set_data(times, energies)
        ax_energy.relim()
        ax_energy.autoscale_view()

        return [im, line_energy]

    # Temperature slider
    ax_temp = plt.axes([0.25, 0.15, 0.65, 0.03])
    # include physical units in the label
    temp_slider = Slider(
        ax_temp,
        "Temp (J/$k_B$)",
        0.1,
        5.0,
        valinit=temp_init,
        valstep=0.1,
    )

    # Critical temperature marker (updates with lattice size)
    tc_current = [critical_temperature(size, J)]
    tc_line = ax_temp.axvline(tc_current[0], color="red", linestyle="--", linewidth=1)
    # place the "Tc" label directly below the marker
    tc_text = ax_temp.text(
        tc_current[0],
        -0.1,
        "Tc",
        transform=ax_temp.get_yaxis_transform(),
        color="red",
        ha="center",
        va="top",
        fontsize=8,
        clip_on=False,
    )

    # Lattice size slider
    ax_size = plt.axes([0.25, 0.2, 0.65, 0.03])
    size_slider = Slider(
        ax_size,
        "Size (sites)",
        10,
        1000,
        valinit=size,
        valstep=10,
    )

    # Speed slider controlling frame delay
    ax_speed = plt.axes([0.25, 0.1, 0.65, 0.03])
    speed_slider = Slider(
        ax_speed,
        "Speed (s)",
        0.01,
        0.5,
        valinit=0.2,
        valstep=0.01,
    )

    def update_speed(val):
        ani.event_source.interval = int(speed_slider.val * 1000)

    def update_size(val):
        nonlocal spins, size
        size = int(size_slider.val)
        spins = initialize_spins(size)
        im.set_data(spins)
        tc_current[0] = critical_temperature(size, J)
        tc_line.set_xdata([tc_current[0], tc_current[0]])
        tc_text.set_x(tc_current[0])

        fig.canvas.draw_idle()

    speed_slider.on_changed(update_speed)
    size_slider.on_changed(update_size)

    # Reset button to randomize spins
    ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
    reset_button = Button(ax_reset, "Reset")

    def reset(event):
        nonlocal spins
        spins = initialize_spins(size)
        im.set_data(spins)
        energies.clear()
        times.clear()
        line_energy.set_data([], [])
        ax_energy.relim()
        ax_energy.autoscale_view()
        fig.canvas.draw_idle()

    reset_button.on_clicked(reset)


    # keep a reference to the animation object so it is not garbage-collected
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=steps,
        interval=int(speed_slider.val * 1000),
        blit=True,
    )

    plt.show()


if __name__ == "__main__":
    main()
