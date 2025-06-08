# Ising Model Simulation

This repository contains a simple Python script that simulates an Ising model on a 50x50 grid. Each lattice site contains a spin (`+1` or `-1`). The script initializes the spins randomly and then evolves the system using the Metropolis algorithm, displaying the lattice as an animation.

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Numba

Install the requirements with:

```bash
pip install numpy matplotlib numba
```

## Running the simulation

Execute the simulation with:

```bash
python ising_model.py
```

Running the script will open a window showing the 2D lattice updating over time. Spins are represented as black and white cells, flipping according to thermal fluctuations.
Each update step pauses for half a second so you can clearly observe the system's evolution.
Numba is used to JIT compile the core routines; the initial call may take a
moment while the compiled code is cached for later runs.
The window also includes interactive controls:

- **Temp slider** – adjust the system temperature in real time. A dashed red
  line marks the critical temperature `Tc` for `kB=1` and `J=1` (~2.27).
- **Size slider** – change the lattice from 10x10 to 1000x1000. The `Tc`
  marker moves to reflect the current size.

- **Speed slider** – set how fast each frame updates.
- **Reset button** – randomize all spins instantly.
- The current **energy** is displayed in the corner of the plot.

## Order parameter curve

The interactive viewer no longer plots magnetization vs. temperature directly.
To generate a proper $|M|(T)$ curve, run the helper script:

```bash
python order_parameter.py
```

It sweeps through a range of temperatures, equilibrates the lattice and
measures both the magnetization and heat capacity at each step. The script
then displays the order-parameter curve together with the heat capacity
as a function of temperature.

Temperatures must be strictly positive. The computation now keeps the spin
configuration from one temperature step to the next and begins with an ordered
state below the critical temperature. This greatly reduces fluctuations in the
measured observables, especially for $T < T_c$.

After collecting the data, the script fits $|M|(T)$ using a piece-wise
sigmoid with separate slopes below and above the transition:
\[\tfrac12\bigl[1+\tanh(a_{\minus}(T_c-T))\bigr]\;\text{for}\;T<T_{split},\quad
\tfrac12\bigl[1+\tanh(a_{\plus}(T_c-T))\bigr]\;\text{for}\;T\ge T_{split}.\]
Optionally, a linear section of configurable width can bridge the two
sigmoids so that the curve remains continuous.  The fitted parameters
$T_c$, $a_{\minus}$, $a_{\plus}$, $T_{split}$ and the width are
printed to the console and shown on the magnetisation plot.  The fitted curve
itself appears as a red line labelled "fit".

## Understanding the critical temperature

The 2‑D Ising Hamiltonian is
\[E=-J\sum_{\langle ij \rangle} s_i s_j,\]
where the spins $s_i=\pm1$ sit on a square lattice and the sum runs over
nearest neighbours.  At low temperature the exchange coupling $J$ aligns the
spins, yielding a non‑zero magnetisation
\(M=\langle |\sum_i s_i|/N\rangle\).  Thermal fluctuations disorder the system
above the critical temperature $T_c$, driving $M$ to zero.  A peak in the heat
capacity marks this transition.

Onsager famously solved the infinite lattice exactly, giving
\[T_c^{\infty}=2J/\ln(1+\sqrt2)\approx2.269\quad(k_B=J=1).\]
Our simulations, however, use a finite grid.  Finite lattices cannot exhibit a
true phase transition, so the magnetisation curve is rounded and the apparent
$T_c$ is shifted slightly.  The helper function `critical_temperature()` applies
a simple finite‑size correction
\(T_c(L)=T_c^{\infty}\,(1-1/L)\).  The order‑parameter script fits an effective
transition temperature to the data it collects, and that fit value approaches
the Onsager result only as the lattice size increases.  Consequently the fitted
$T_c$ is close to but not exactly the critical temperature of an infinite
lattice.

