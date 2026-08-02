r"""Experiment 6: the gauge field as a function on context space.

Discrete context labels can only ask whether a transformation was fitted for
each recorded condition.  The stronger question is whether the model learns a
*field* over context space -- a map ``c -> T_c`` that is defined, and correct, at
cue values never seen during training.  That is what a gauge connection is, and
it is what separates the framework from per-condition alignment.

**Part A: interpolation.**  Thirteen cue-card orientations spanning a
120-degree arc, with the model's context variable being the observable cue
direction ``(cos alpha, sin alpha)``.  Four orientations are held out entirely,
and the activity recorded in them is then predicted from the cue value alone.
Per-condition alignment methods cannot be evaluated here at all -- they have no
transformation for an unobserved condition -- so we compare against the two
alternatives that *are* well defined: leaving the latent untransformed, and
re-using the transformation of the nearest observed cue angle.

**Part B: a topological obstruction.**  When the cue family closes into a full
circle, the required group elements ``R(alpha)`` trace a loop of winding number
one in ``GL(d)^+``.  But ``T_c = exp(sum_k theta_k(c) G_k)`` with a continuous
``theta`` is null-homotopic for *any* ``theta``, because ``theta`` factors
through a contractible vector space.  A single-chart gauge field therefore
cannot cover the circle -- not as a matter of optimisation but as a matter of
topology.  We verify this by fitting the same architecture on the circle and on
its universal cover (the unwrapped angle), which differ in nothing else.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ..geometry.manifold import apply_readout, fit_latent_readout
from ..geometry.metrics import GaugeSummary, r2_score_matrix
from ..models.baselines import RandomFeatureReadout
from ..models.gnd import GNDConfig
from ..models.train import fit_gnd
from ..simulations.hippocampus import simulate_place_cells
from ..utils.common import RESULTS_DIR, aggregate_seeds, save_json, set_seed
from .common import checkpoint
from .evaluate import canonical_from_model, gauge_summary_from_model

ARC_HELD_OUT = (1, 4, 8, 11)          # indices within the 13-point arc


def _theta_at(model, feats: np.ndarray, ref_feats: np.ndarray, device="cpu") -> np.ndarray:
    """Query the learned gauge field at arbitrary context variables."""
    with torch.no_grad():
        f = torch.as_tensor(np.atleast_2d(feats), dtype=torch.float32, device=device)
        r = torch.as_tensor(ref_feats, dtype=torch.float32, device=device)
        return model.coefficients(f, r).cpu().numpy()


def _matrices(model, theta: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.gauge.matrices(torch.as_tensor(theta, dtype=torch.float32)).cpu().numpy()


def _fit_and_probe(full, train_ctx, cfg, device, verbose=False):
    """Fit on a subset of contexts, then evaluate at *every* context."""
    seen = full.select_contexts(train_ctx)
    tr, _ = seen.split(0.8, seed=cfg.seed)
    _, te = full.split(0.8, seed=cfg.seed)

    model, hist = fit_gnd(tr, cfg, device=device, verbose=verbose)
    ref_feats = seen.context_features[seen.reference]
    theta_all = _theta_at(model, full.context_features, ref_feats, device)
    M_all = _matrices(model, theta_all)

    w_tr, _ = canonical_from_model(model, tr, device)
    w_te, _ = canonical_from_model(model, te, device)
    theta_ref = _theta_at(model, ref_feats[None], ref_feats, device)[0]
    ref_idx = full.reference
    with torch.no_grad():
        zref = model.gauge.inverse(
            torch.as_tensor(w_te[:, ref_idx], dtype=torch.float32),
            torch.as_tensor(theta_ref, dtype=torch.float32).expand(len(w_te), -1),
        ).cpu().numpy()

    readout_map = RandomFeatureReadout(n_features=768, ridge=1e-3, seed=cfg.seed).fit(
        w_tr.reshape(-1, w_tr.shape[-1]), tr.activity.reshape(-1, tr.n_neurons)
    )
    gauge = GaugeSummary(
        generators=gauge_summary_from_model(model, seen, device).generators,
        coefficients=theta_all, matrices=M_all, affine=model.gauge.affine,
    )
    chart = fit_latent_readout(zref, te.latent)
    return model, gauge, zref, te, readout_map, chart, ref_feats, hist


def _score_contexts(full, te, gauge, zref, readout_map, chart, train_ctx, tag, seed):
    angles = np.array([s.group_params["theta"] for s in full.contexts])
    rows = []
    base_id = readout_map.predict(zref)
    for c in range(full.n_contexts):
        if c == full.reference:
            # T*_ref is the identity, so the GRE denominator vanishes and the
            # transport task is trivial; the reference context carries no
            # information about recovery and is excluded throughout.
            continue
        wc = gauge.apply(zref, c)
        u_true = full.contexts[c].apply(te.latent)
        den = np.sqrt(((u_true - te.latent) ** 2).sum(1).mean()) + 1e-12
        d = np.abs(np.angle(np.exp(1j * (angles[list(train_ctx)] - angles[c]))))
        nn = list(train_ctx)[int(np.argmin(d))]
        rows.append({
            "part": tag, "seed": seed, "context": c,
            "is_reference": False,
            "held_out": c not in train_ctx,
            "angle_deg": float(np.degrees(angles[c])),
            "transport_r2": r2_score_matrix(readout_map.predict(wc), te.activity[:, c]),
            "gre": float(np.sqrt(((apply_readout(chart, wc) - u_true) ** 2).sum(1).mean()) / den),
            "transport_r2_identity": r2_score_matrix(base_id, te.activity[:, c]),
            "transport_r2_nearest_seen": r2_score_matrix(
                readout_map.predict(gauge.apply(zref, nn)), te.activity[:, c]),
            "theta_norm": float(np.linalg.norm(gauge.coefficients[c])),
        })
    return rows


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-cells", type=int, default=120)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--noise", type=float, default=0.15)
    ap.add_argument("--n-latent", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--arc-points", type=int, default=13)
    ap.add_argument("--arc-span-deg", type=float, default=120.0)
    ap.add_argument("--circle-points", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(RESULTS_DIR / "exp6_continuous_context"))
    ap.add_argument("--parts", nargs="*", default=["arc", "circle"])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.seeds, args.n_samples, args.epochs, args.n_cells = [0], 900, 80, 60

    base = GNDConfig(
        n_latent=args.n_latent, hidden=192, depth=3, n_generators=6, algebra="gl",
        epochs=args.epochs, batch_size=256, lr=2e-3,
        w_group=0.2, w_closure=0.05, w_topology=0.5,
    )

    rows, arts = [], {}
    for seed in args.seeds:
        cfg = replace(base, seed=seed)

        # ---------------- Part A: interpolation over a contractible arc ----
        checkpoint(Path(args.out), {"rows": rows, "summary": {}, "args": vars(args),
                                    "complete": False})
        if "arc" in args.parts:
            print(f"=== exp6 [arc] seed {seed} ===")
            set_seed(seed)
            full = simulate_place_cells(
                n_cells=args.n_cells, n_samples=args.n_samples, noise=args.noise,
                context_mode="rotation_family", n_rotations=args.arc_points,
                context_span=np.radians(args.arc_span_deg), context_feature_mode="circular",
                seed=seed,
            ).standardise()
            train_ctx = [c for c in range(full.n_contexts) if c not in ARC_HELD_OUT]
            model, gauge, zref, te, rmap, chart, ref_feats, _ = _fit_and_probe(
                full, train_ctx, cfg, args.device, verbose=(seed == args.seeds[0])
            )
            r = _score_contexts(full, te, gauge, zref, rmap, chart, train_ctx, "arc", seed)
            rows += r
            for flag, lbl in ((False, "trained "), (True, "held-out")):
                sub = [x for x in r if x["held_out"] == flag]
                print(f"  {lbl} cues: transport R2 = {np.mean([x['transport_r2'] for x in sub]):+.3f}  "
                      f"GRE = {np.mean([x['gre'] for x in sub]):.3f}  "
                      f"| identity {np.mean([x['transport_r2_identity'] for x in sub]):+.3f}  "
                      f"nearest-seen {np.mean([x['transport_r2_nearest_seen'] for x in sub]):+.3f}")
            if not arts:
                dense = np.radians(np.linspace(-args.arc_span_deg / 2, args.arc_span_deg / 2, 61))
                dfeat = np.stack([np.cos(dense), np.sin(dense)], axis=1)
                arts = {
                    "arc_dense_alpha": dense,
                    "arc_dense_theta": _theta_at(model, dfeat, ref_feats, args.device),
                    "arc_theta": gauge.coefficients,
                    "arc_angles": np.array([s.group_params["theta"] for s in full.contexts]),
                    "arc_held_out": np.array(ARC_HELD_OUT),
                    "arc_z_ref": zref[:2000],
                    "arc_latent": te.latent[:2000],
                }

        # ---------------- Part B: the winding obstruction -------------------
        if "circle" in args.parts:
            for fmode in ("circular", "lifted"):
                print(f"=== exp6 [circle/{fmode}] seed {seed} ===")
                set_seed(seed)
                full = simulate_place_cells(
                    n_cells=args.n_cells, n_samples=args.n_samples, noise=args.noise,
                    context_mode="rotation_family", n_rotations=args.circle_points,
                    context_span=2 * np.pi, context_feature_mode=fmode, seed=seed,
                ).standardise()
                train_ctx = list(range(full.n_contexts))     # all contexts observed
                model, gauge, zref, te, rmap, chart, ref_feats, _ = _fit_and_probe(
                    full, train_ctx, cfg, args.device
                )
                r = _score_contexts(full, te, gauge, zref, rmap, chart, train_ctx,
                                    f"circle:{fmode}", seed)
                rows += r
                print(f"  transport R2 = {np.mean([x['transport_r2'] for x in r]):+.3f}   "
                      f"GRE = {np.mean([x['gre'] for x in r]):.3f}   "
                      f"max |theta| = {max(x['theta_norm'] for x in r):.2f}")
                if fmode == "circular" and "circle_theta" not in arts:
                    arts["circle_theta"] = gauge.coefficients
                    arts["circle_angles"] = np.array([s.group_params["theta"] for s in full.contexts])
                if fmode == "lifted" and "lifted_theta" not in arts:
                    arts["lifted_theta"] = gauge.coefficients

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for part in sorted({r["part"] for r in rows}):
        if part == "arc":
            for flag, lbl in ((False, "arc_trained_cues"), (True, "arc_held_out_cues")):
                summary[lbl] = aggregate_seeds(
                    [r for r in rows if r["part"] == "arc" and r["held_out"] == flag]
                )
        else:
            summary[part.replace(":", "_")] = aggregate_seeds([r for r in rows if r["part"] == part])
    save_json({"rows": rows, "summary": summary, "args": vars(args), "complete": True},
              out / "results.json")
    if arts:
        # Stamped with the seed, so the archive cannot outlive the run silently.
        np.savez_compressed(out / "artifacts.npz",
                            provenance_seed=np.array(args.seeds[0]), **arts)
    print(f"\nwrote {out/'results.json'}")
    return {"rows": rows, "summary": summary}


if __name__ == "__main__":
    main()
