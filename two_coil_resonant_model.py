"""Two-coil resonant wireless power transfer model (Tesla-style coupled resonators).

Geometry -> inductance L and mutual inductance M -> coupling k -> circuit solution.

Circuit: series-resonant transmitter and receiver (source -> Rs, C1, L1 || L2, C2 -> RL).
Coils are coaxial N-turn circular loops treated as a single filament bundle.

Run:  python two_coil_resonant_model.py            (prints a summary, writes PNG plots)
      python two_coil_resonant_model.py --no-plot  (summary and self-checks only)
"""
from dataclasses import dataclass
import sys

import numpy as np
from scipy.special import ellipk, ellipe

MU0 = 4e-7 * np.pi


@dataclass
class Coil:
    radius: float      # m, mean loop radius
    turns: int
    wire_radius: float  # m
    q: float           # unloaded quality factor at the design frequency

    @property
    def inductance(self):
        # Single circular loop: L = mu0*a*(ln(8a/r_w) - 2); turns scale as N^2.
        # Tight-bundle approximation; real multi-turn coils need Wheeler/Nagaoka corrections.
        return self.turns**2 * MU0 * self.radius * (np.log(8 * self.radius / self.wire_radius) - 2)


def mutual_inductance(c1: Coil, c2: Coil, d):
    """Maxwell's formula for coaxial circular filaments, scaled by N1*N2. d in metres (array ok)."""
    a, b = c1.radius, c2.radius
    d = np.asarray(d, dtype=float)
    m2 = 4 * a * b / ((a + b) ** 2 + d**2)
    m = np.sqrt(m2)
    m_single = MU0 * np.sqrt(a * b) * ((2 / m - m) * ellipk(m2) - (2 / m) * ellipe(m2))
    return c1.turns * c2.turns * m_single


def coupling(c1, c2, d):
    return mutual_inductance(c1, c2, d) / np.sqrt(c1.inductance * c2.inductance)


def solve_link(c1, c2, d, f, f0, rl, rs=0.0, c1_tune=None, c2_tune=None):
    """Solve the series-series link at frequency f (Hz). Returns dict of efficiencies and currents.

    Capacitors are tuned to resonate each coil at f0 unless overridden. Source amplitude is 1 V peak.
    Coil ESR is R = w0*L/Q evaluated at f0 (constant with frequency: skin effect ignored).
    """
    w, w0 = 2 * np.pi * np.asarray(f, dtype=float), 2 * np.pi * f0
    L1, L2 = c1.inductance, c2.inductance
    R1, R2 = w0 * L1 / c1.q, w0 * L2 / c2.q
    C1 = c1_tune if c1_tune else 1 / (w0**2 * L1)
    C2 = c2_tune if c2_tune else 1 / (w0**2 * L2)
    M = mutual_inductance(c1, c2, d)

    Z1 = rs + R1 + 1j * w * L1 + 1 / (1j * w * C1)
    Z2 = R2 + rl + 1j * w * L2 + 1 / (1j * w * C2)
    I1 = 1.0 * Z2 / (Z1 * Z2 + (w * M) ** 2)          # Vs = 1 V
    I2 = -1j * w * M * I1 / Z2
    p_load = 0.5 * rl * np.abs(I2) ** 2
    p_src = 0.5 * np.real(I1)                           # Vs=1 real -> Re(Vs I1*)/2
    p_link_in = p_src - 0.5 * rs * np.abs(I1) ** 2      # power entering the TX coil circuit
    return {
        "eta_total": p_load / p_src,
        "eta_link": p_load / p_link_in,
        "i1": I1, "i2": I2, "p_load": p_load, "k": M / np.sqrt(L1 * L2),
    }


def eta_max_theory(k, q1, q2):
    """Optimal-load maximum link efficiency: x/(1+sqrt(1+x))^2, x = k^2 Q1 Q2."""
    x = k**2 * q1 * q2
    return x / (1 + np.sqrt(1 + x)) ** 2


def optimal_load(c1, c2, d, f0):
    """Load resistance that maximises link efficiency at f0 (series-series)."""
    w0 = 2 * np.pi * f0
    R2 = w0 * c2.inductance / c2.q
    R1 = w0 * c1.inductance / c1.q
    M = mutual_inductance(c1, c2, d)
    return R2 * np.sqrt(1 + (w0 * M) ** 2 / (R1 * R2))


def self_check():
    """Sanity checks: model must agree with the analytic optimum and limiting behaviour."""
    c = Coil(radius=0.15, turns=10, wire_radius=1e-3, q=300)
    f0 = 1.0e6
    for d in (0.05, 0.15, 0.30, 0.60):
        k = float(coupling(c, c, d))
        rl = float(optimal_load(c, c, d, f0))
        eta = float(solve_link(c, c, d, f0, f0, rl)["eta_link"])
        theory = float(eta_max_theory(k, c.q, c.q))
        assert abs(eta - theory) < 1e-6, (d, eta, theory)
    # k must fall monotonically with distance and approach 1 only at tiny separation
    ks = coupling(c, c, np.array([0.01, 0.1, 0.3, 1.0]))
    assert np.all(np.diff(ks) < 0) and ks[0] < 1
    # far field: M ~ 1/d^3 for d >> radius
    r = coupling(c, c, 4.0) / coupling(c, c, 8.0)
    assert 7 < r < 9, r
    print("self-checks passed")


def main(plot=True):
    self_check()
    # Example rig: two identical 30 cm diameter, 10-turn coils, 2 mm wire, Q=300, 1 MHz
    coil = Coil(radius=0.15, turns=10, wire_radius=1e-3, q=300)
    f0 = 1.0e6
    print(f"L = {coil.inductance*1e6:.1f} uH, C(resonant) = {1/((2*np.pi*f0)**2*coil.inductance)*1e12:.0f} pF")
    print(f"{'d (m)':>6} {'d/D':>5} {'k':>8} {'R_opt (ohm)':>12} {'eta_link':>9}")
    dist = np.array([0.05, 0.1, 0.15, 0.3, 0.45, 0.6, 0.9, 1.2])
    for d in dist:
        rl = float(optimal_load(coil, coil, d, f0))
        res = solve_link(coil, coil, d, f0, f0, rl)
        print(f"{d:6.2f} {d/(2*coil.radius):5.1f} {float(res['k']):8.4f} {rl:12.2f} {float(res['eta_link']):9.3f}")

    if not plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    dd = np.linspace(0.03, 1.5, 300)
    for q in (100, 300, 1000):
        cq = Coil(coil.radius, coil.turns, coil.wire_radius, q)
        eta = [float(solve_link(cq, cq, d, f0, f0, float(optimal_load(cq, cq, d, f0)))["eta_link"]) for d in dd]
        ax[0].plot(dd / (2 * cq.radius), eta, label=f"Q = {q}")
    ax[0].set(xlabel="separation / coil diameter", ylabel="max link efficiency",
              title="Efficiency vs distance (optimal load)", ylim=(0, 1))
    ax[0].grid(alpha=.3)
    ax[0].legend()

    ax[1].semilogy(dd / (2 * coil.radius), coupling(coil, coil, dd))
    ax[1].set(xlabel="separation / coil diameter", ylabel="coupling k", title="Coupling coefficient")
    ax[1].grid(alpha=.3, which="both")

    f = np.linspace(0.8e6, 1.2e6, 1500)
    for d in (0.1, 0.2, 0.4, 0.8):
        rl = float(optimal_load(coil, coil, 0.4, f0))  # fixed load across the sweep
        ax[2].plot(f / 1e6, solve_link(coil, coil, d, f, f0, rl)["eta_link"],
                   label=f"d = {d} m (k={float(coupling(coil, coil, d)):.3f})")
    ax[2].set(xlabel="frequency (MHz)", ylabel="link efficiency",
              title="Frequency response (fixed load): splitting when over-coupled")
    ax[2].grid(alpha=.3)
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("two_coil_resonant_model.png", dpi=130)
    print("wrote two_coil_resonant_model.png")


if __name__ == "__main__":
    main(plot="--no-plot" not in sys.argv)
