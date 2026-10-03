"""Fit the two-coil resonant model to bench measurements.

Input CSV (header required):  distance_m, eta [, freq_hz]
  distance_m  coil-to-coil separation (m, centre to centre)
  eta         measured efficiency in 0..1  (P_load / P_in, see --eta-kind)
  freq_hz     optional drive frequency for that row (default: --f0)

Fitted parameters: Q of each coil and k_scale (multiplier on the computed mutual inductance,
absorbing winding/geometry error). Coil radius, turns, wire radius, f0, source and load
resistance are inputs you measure directly.

Usage:
  python rig_fit.py --data rig.csv --radius 0.15 --turns 10 --wire 1e-3 --f0 1e6 --rs 50 --rl 50
  python rig_fit.py --demo        # synthetic self-test: generates fake data and checks recovery
"""
import argparse

import numpy as np
from scipy.optimize import least_squares

from two_coil_resonant_model import Coil, solve_link, coupling


def model_eta(params, d, f, geom, f0, rs, rl, kind):
    q1, q2, ks = np.exp(params[0]), np.exp(params[1]), params[2]
    c1 = Coil(geom["radius"], geom["turns"], geom["wire"], q1)
    c2 = Coil(geom["radius"], geom["turns"], geom["wire"], q2)
    out = np.empty_like(d)
    for i, (di, fi) in enumerate(zip(d, f)):
        out[i] = float(solve_link(c1, c2, di, fi, f0, rl, rs=rs, k_scale=ks)[
            "eta_total" if kind == "total" else "eta_link"])
    return out


def fit(d, eta, f, geom, f0, rs, rl, kind="total", q_guess=200.0, tie_q=True):
    """Least-squares fit. tie_q=True fits one shared Q (the geometric mean sqrt(Q_tx*Q_rx)):
    efficiency-vs-distance data cannot separate Q_tx from Q_rx and k_scale, so separate Qs
    need extra data (e.g. a frequency sweep or a measured Q per coil from a ring-down)."""
    def expand(p):
        return np.array([p[0], p[0], p[1]]) if tie_q else np.asarray(p)

    def resid(p):
        return model_eta(expand(p), d, f, geom, f0, rs, rl, kind) - eta

    if tie_q:
        x0, lo, hi, xs = [np.log(q_guess), 1.0], [np.log(5), 0.3], [np.log(5000), 3.0], [1, 0.3]
    else:
        x0 = [np.log(q_guess), np.log(q_guess), 1.0]
        lo, hi, xs = [np.log(5), np.log(5), 0.3], [np.log(5000), np.log(5000), 3.0], [1, 1, 0.3]
    sol = least_squares(resid, np.array(x0), bounds=(lo, hi), x_scale=xs)
    npar = len(x0)
    dof = max(len(d) - npar, 1)
    s2 = np.sum(sol.fun**2) / dof
    try:
        jtj = sol.jac.T @ sol.jac
        err = np.sqrt(np.diag(np.linalg.inv(jtj) * s2))
        cond = np.linalg.cond(jtj)
    except np.linalg.LinAlgError:
        err, cond = np.full(npar, np.nan), np.inf
    sol.x = expand(sol.x)                       # always report as [lnQtx, lnQrx, k_scale]
    err = np.array([err[0], err[0], err[1]]) if tie_q else err
    return sol, err, cond, np.sqrt(s2)


def report(sol, err, cond, rms):
    q1, q2, ks = np.exp(sol.x[0]), np.exp(sol.x[1]), sol.x[2]
    if np.isclose(sol.x[0], sol.x[1]):
        print(f"Q (shared, = sqrt(Q_tx*Q_rx)) = {q1:8.1f}  (+/- {q1*err[0]:.1f})")
    else:
        print(f"Q_tx    = {q1:8.1f}  (+/- {q1*err[0]:.1f})")
        print(f"Q_rx    = {q2:8.1f}  (+/- {q2*err[1]:.1f})")
    print(f"k_scale = {ks:8.3f}  (+/- {err[2]:.3f})   (1.0 = computed geometry is exact)")
    print(f"rms residual = {rms:.4f} efficiency units")
    if cond > 1e8 or np.any(err[:2] > 0.5):
        print("WARNING: parameters poorly constrained (Q_tx, Q_rx and k_scale are partly degenerate). "
              "Add points at several separations, including d > 1 coil diameter, and ideally "
              "a frequency sweep at one separation.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--radius", type=float)
    ap.add_argument("--turns", type=int)
    ap.add_argument("--wire", type=float, default=1e-3, help="wire radius (m)")
    ap.add_argument("--f0", type=float, help="resonant/drive frequency (Hz)")
    ap.add_argument("--rs", type=float, default=0.0, help="source resistance (ohm)")
    ap.add_argument("--rl", type=float, help="load resistance (ohm)")
    ap.add_argument("--eta-kind", choices=["total", "link"], default="total",
                    help="total: P_load/P_generator ; link: P_load/P_into_TX_coil")
    ap.add_argument("--separate-q", action="store_true",
                    help="fit Q_tx and Q_rx separately (needs data that can separate them)")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    if a.demo:
        geom = {"radius": 0.15, "turns": 10, "wire": 1e-3}
        f0, rs, rl = 1e6, 5.0, 20.0
        rng = np.random.default_rng(0)
        d = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8])
        truth = np.array([np.log(240.0), np.log(310.0), 0.85])
        eta = model_eta(truth, d, np.full_like(d, f0), geom, f0, rs, rl, "total")
        eta = np.clip(eta + rng.normal(0, 0.01, d.shape), 0, 1)
        print("SYNTHETIC data (not your rig): truth Q_tx=240, Q_rx=310 (geometric mean 273), "
              "k_scale=0.85, 1% noise")
        sol, err, cond, rms = fit(d, eta, np.full_like(d, f0), geom, f0, rs, rl)
        report(sol, err, cond, rms)
        return

    if not (a.data and a.radius and a.turns and a.f0 and a.rl):
        ap.error("--data, --radius, --turns, --f0 and --rl are required (or use --demo)")
    tab = np.genfromtxt(a.data, delimiter=",", names=True)
    d, eta = tab["distance_m"], tab["eta"]
    f = tab["freq_hz"] if "freq_hz" in tab.dtype.names else np.full_like(d, a.f0)
    geom = {"radius": a.radius, "turns": a.turns, "wire": a.wire}
    sol, err, cond, rms = fit(d, eta, f, geom, a.f0, a.rs, a.rl, a.eta_kind, tie_q=not a.separate_q)
    report(sol, err, cond, rms)
    print(f"computed k at each distance (k_scale applied): "
          f"{[round(float(coupling(Coil(a.radius, a.turns, a.wire, 1), Coil(a.radius, a.turns, a.wire, 1), x))*sol.x[2], 4) for x in d]}")


if __name__ == "__main__":
    main()
