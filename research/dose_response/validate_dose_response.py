"""Read-only campaign audit and synthetic validation of nonmonotonic responses.

Run from the repository root with Python and the project's optional BoTorch deps.
Writes only to the chosen report directory; never invokes campaign selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cryo-dose-mpl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/08_multi_objective"))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from helper.batch_gp import BatchGP, torch_bundle
from helper.gp_strategy import viability_rows, fit_viability, shared_scores, snapshot_current
from helper.explainability import load_round_evidence
from helper.gp_comparison import attach_roles
from helper.registry import load_registry
from helper.models import build_training_frame, paired_objective_frame, _fit_regression
from helper.group10_config import production_observations

REG = load_registry(ROOT / "config_v2/ingredients.yaml")
FEATURES = REG.feature_names
E = FEATURES.index("ectoin_M")
SEED = 20261022


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe(x, omit=None):
    return "; ".join(f"{f}={v:.6g}" for f, v in zip(FEATURES, x) if v > 1e-6 and f != omit) or "all other modeled ingredients zero"


def curve(model, background, feature, doses, seed=SEED, draws=4096, delta=5.):
    j = FEATURES.index(feature)
    x = np.tile(background, (len(doses), 1)); x[:, j] = doses
    mean, cov = model.posterior(x)
    sd = np.sqrt(np.maximum(np.diag(cov), 0))
    k = int(np.argmax(mean)); inside = 0 < k < len(doses)-1
    # Joint draws preserve covariance between the low, middle and high doses.
    selected = [0, k, len(doses)-1]
    c = cov[np.ix_(selected, selected)]
    w, v = np.linalg.eigh(c)
    samples = mean[selected] + np.random.default_rng(seed).normal(size=(draws, 3)) @ (v * np.sqrt(np.maximum(w, 0))).T
    contrast = np.minimum(samples[:, 1]-samples[:, 0], samples[:, 1]-samples[:, 2])
    summary = dict(feature=feature, peak_dose=float(doses[k]), low_dose=float(doses[0]), high_dose=float(doses[-1]),
                   low_mean=float(mean[0]), peak_mean=float(mean[k]), high_mean=float(mean[-1]),
                   gain_vs_low=float(mean[k]-mean[0]), gain_vs_high=float(mean[k]-mean[-1]),
                   interior_mean_peak=inside, probability_both_gains_positive=float(np.mean(contrast > 0)) if inside else 0.,
                   probability_both_gains_over_delta=float(np.mean(contrast > delta)) if inside else 0.,
                   delta=delta, contrast_lower=float(np.quantile(contrast, .025)), contrast_upper=float(np.quantile(contrast, .975)))
    table = pd.DataFrame(dict(dose=doses, mean=mean, sd=sd, lower=mean-1.96*sd, upper=mean+1.96*sd))
    return summary, table


def actual(out):
    print("Auditing frozen campaign posterior through Round 14", flush=True)
    forms = pd.read_csv(ROOT / "data/processed_v2/formulations.csv")
    obs = pd.read_csv(ROOT / "data/processed_v2/observations.csv")
    prior, active, info = load_round_evidence(ROOT, 15, obs, FEATURES)
    assert active is not None and active.strategy == "control"
    prior = attach_roles(prior, ROOT)
    rows = viability_rows(forms, prior, FEATURES)
    ordinary = rows.loc[~rows.is_control].copy()
    assert len(ordinary) == (~active.controls).sum()
    assert np.allclose(ordinary[FEATURES].to_numpy(), active.x[~active.controls])
    ect = ordinary.loc[ordinary.ectoin_M > 1e-6].copy()
    ect["background"] = [recipe(x, "ectoin_M") for x in ect[FEATURES].to_numpy()]
    ect.to_csv(out / "ectoine_observations.csv", index=False)
    # Primary context chosen by repeat count, without reference to response.
    sentinel = ect.formulation_id.value_counts().sort_values(ascending=False).index[0]
    bg = ect.loc[ect.formulation_id.eq(sentinel), FEATURES].iloc[0].to_numpy(float)
    contexts = {"most_repeated_ectoine_recipe": bg, "EG_0.5M": np.zeros(len(FEATURES)),
                "EG_1M_FBS_5pct": np.zeros(len(FEATURES))}
    contexts["EG_0.5M"][FEATURES.index("ethylene_glycol_M")] = .5
    contexts["EG_1M_FBS_5pct"][FEATURES.index("ethylene_glycol_M")] = 1.
    contexts["EG_1M_FBS_5pct"][FEATURES.index("fbs_pct")] = 5.
    doses = np.linspace(0, .5, 101)
    summaries, curves = [], []
    frozen = ROOT / "results/multi_objective_v2/rounds/ROUND_015/gp_comparison"
    manifest = json.loads((frozen / "manifest.json").read_text())
    for strategy in manifest["models"]:
        assert sha(frozen / f"{strategy}.json") == manifest["hashes"][f"{strategy}.json"]
        model = BatchGP.load(frozen / f"{strategy}.json")
        for context, background in contexts.items():
            s, t = curve(model, background, "ectoin_M", doses)
            summaries.append(dict(strategy=strategy, context=context, background=recipe(background, "ectoin_M"), **s))
            curves.append(t.assign(strategy=strategy, context=context))
    pd.DataFrame(summaries).to_csv(out / "ectoine_strategy_contrasts.csv", index=False)
    pd.concat(curves).to_csv(out / "ectoine_curves.csv", index=False)
    # Full formulation covariance checks: NumPy production posterior vs acquisition adapter.
    import torch
    torch.set_num_threads(1)
    xcheck = np.tile(bg, (7, 1)); xcheck[:, E] = np.linspace(0, .5, 7)
    p = torch_bundle([active]).posterior(torch.tensor(xcheck, dtype=torch.double))
    pm, pc = active.posterior(xcheck)
    assert np.allclose(p.mean.detach().numpy().ravel(), pm, atol=1e-8)
    assert np.allclose(p.distribution.covariance_matrix.detach().numpy(), pc, atol=1e-6)
    # Comparable mechanical objective, same reconstruction as production.
    frame = build_training_frame(forms, production_observations(prior, target_round_number=15), REG)
    endpoint = "critical_axial_load_N_per_needle"
    mf = frame.dropna(subset=[endpoint]).copy()
    ms = _fit_regression(mf[FEATURES].to_numpy(float), mf[endpoint],
                         y_noise=mf.get(endpoint + "__noise", pd.Series(index=mf.index, dtype=float)))
    mechanical = snapshot_current(ms, mf[FEATURES].to_numpy(float), mf[endpoint].to_numpy(float), mf.batch_id.to_numpy())
    bundle = SimpleNamespace(acquisition_bundle=torch_bundle([active, mechanical]), gp_strategy_decision={"viability_strategy":"control"})
    paired = paired_objective_frame(frame).dropna(subset=["viability_percent", endpoint])
    acquisition, mechanical_summaries = [], []
    for context, background in contexts.items():
        x = np.tile(background, (len(doses), 1)); x[:, E] = doses
        scores, _ = shared_scores(bundle, paired[FEATURES].to_numpy(float), x, (0., 0.))
        mmean, mcov = mechanical.posterior(x)
        mechanical_summary, _ = curve(mechanical, background, "ectoin_M", doses, delta=.01)
        mechanical_summaries.append(dict(context=context, **mechanical_summary))
        acquisition.append(pd.DataFrame(dict(context=context, dose=doses, qlognehvi=scores,
                               mechanics_mean=mmean, mechanics_sd=np.sqrt(np.maximum(np.diag(mcov), 0)))))
    pd.concat(acquisition).to_csv(out / "actual_raw_acquisition.csv", index=False)
    pd.DataFrame(mechanical_summaries).to_csv(out / "ectoine_mechanical_contrasts.csv", index=False)
    # Dose support and an outcome-independent representative background for every ingredient.
    scan, coverage, all_contexts = [], [], []
    for j, feature in enumerate(FEATURES):
        present = ordinary.loc[ordinary[feature] > 1e-6].drop_duplicates("formulation_id")
        others = [f for f in FEATURES if f != feature]
        groups = ordinary.groupby(others, dropna=False)[feature].nunique()
        coverage.append(dict(feature=feature, positive_formulations=len(present), positive_doses=present[feature].nunique(),
                             exact_backgrounds_with_2_doses=int((groups >= 2).sum()), exact_backgrounds_with_3_doses=int((groups >= 3).sum())))
        if present[feature].nunique() < 3:
            continue
        z = present[FEATURES].to_numpy(float) / active.scale
        z[:, j] = 0
        medoid = int(np.argmin(cdist(z,z).sum(axis=1)))
        background = present.iloc[medoid][FEATURES].to_numpy(float)
        high = min(float(present[feature].max()), REG.get_by_feature(feature).practical_upper_bound or REG.get_by_feature(feature).upper_bound)
        s, _ = curve(active, background, feature, np.linspace(0, high, 101))
        scan.append(dict(background=recipe(background, feature), background_formulation_id=present.iloc[medoid].formulation_id, **s))
    for fid, group in ect.groupby("formulation_id"):
        background = group.iloc[0][FEATURES].to_numpy(float)
        s, _ = curve(active, background, "ectoin_M", doses)
        all_contexts.append(dict(formulation_id=fid, background=recipe(background, "ectoin_M"), **s))
    pd.DataFrame(scan).to_csv(out / "ingredient_screen.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "dose_support.csv", index=False)
    pd.DataFrame(all_contexts).to_csv(out / "ectoine_all_observed_backgrounds.csv", index=False)
    history = []
    for target in range(11, 16):
        _, historical, _ = load_round_evidence(ROOT, target, obs, FEATURES)
        for context, background in contexts.items():
            s, _ = curve(historical, background, "ectoin_M", doses)
            history.append(dict(data_through_round=target-1, context=context, **s))
    pd.DataFrame(history).to_csv(out / "ectoine_history.csv", index=False)
    # Full refits after removing entire rounds, including that round's controls.
    sensitivity, heldout = [], []
    for batch in sorted(ordinary.batch_id.unique()):
        model = fit_viability(forms, prior.loc[prior.batch_id.ne(batch)], FEATURES, "control")
        for context, background in contexts.items():
            s, _ = curve(model, background, "ectoin_M", doses)
            sensitivity.append(dict(omitted_batch=batch, context=context, **s))
        test = ordinary.loc[ordinary.batch_id.eq(batch)]
        pred, cov = model.posterior(test[FEATURES].to_numpy(float))
        future_sd = np.sqrt(np.maximum(np.diag(cov),0) + model.batch_sd**2 + model.noise_sd**2)
        heldout.append(pd.DataFrame(dict(formulation_id=test.formulation_id, batch_id=batch, ectoin_M=test.ectoin_M,
                           actual=test.value, predicted=pred, future_sd=future_sd)))
    pd.DataFrame(sensitivity).to_csv(out / "leave_round_out_shape.csv", index=False)
    ho = pd.concat(heldout); ho.to_csv(out / "leave_round_out_predictions.csv", index=False)
    # Exact background-matched titration is absent: this holdout cannot prove causal dose shape.
    h = ho.loc[ho.ectoin_M > 1e-6]
    sources = [ROOT / "data/processed_v2/formulations.csv", ROOT / "data/processed_v2/observations.csv",
               frozen / "training_observations.csv", frozen / "manifest.json", Path(__file__)]
    sources += list((ROOT / "src/08_multi_objective/helper").glob("batch_gp.py"))
    sources += [ROOT / "src/08_multi_objective/helper/gp_strategy.py", ROOT / "src/08_multi_objective/helper/models.py"]
    stats = dict(training_ordinary=len(ordinary), training_controls=int(rows.is_control.sum()), ectoine_rows=len(ect),
                 ectoine_unique_formulations=ect.formulation_id.nunique(), mechanical_rows=len(mf), paired_baseline_rows=len(paired),
                 primary_formulation_id=sentinel, contexts={k: recipe(v,"ectoin_M") for k,v in contexts.items()},
                 batch_sd=active.batch_sd, residual_sd=active.noise_sd, ectoine_lengthscale_M=active.scale[E],
                 ectoine_loro_rmse=float(np.sqrt(np.mean((h.actual-h.predicted)**2))),
                 ectoine_loro_95_coverage=float(np.mean(abs(h.actual-h.predicted) <= 1.96*h.future_sd)),
                 source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources}, provenance=info,
                 posterior_adapter_verified=True)
    save_json(out / "actual_summary.json", stats)
    print(json.dumps({k:v for k,v in stats.items() if k not in ("source_hashes", "provenance", "contexts")}), flush=True)


def true_response(d, kind="broad"):
    if kind == "monotone":
        return 20 + 110*d
    width = .025 if kind == "narrow" else .10
    return 20 + 65*np.exp(-.5*((d-.24)/width)**2)


def synthetic(out, seeds):
    print(f"Synthetic recovery and negative controls: {seeds} seeds per condition", flush=True)
    forms = pd.read_csv(ROOT / "data/processed_v2/formulations.csv")
    obs = attach_roles(pd.read_csv(ROOT / "results/multi_objective_v2/rounds/ROUND_015/gp_comparison/training_observations.csv"), ROOT)
    real = viability_rows(forms, obs, FEATURES)
    results, examples = [], []
    settings = [("bracketed", "broad"), ("one_sided", "broad"), ("bracketed", "narrow"),
                ("bracketed", "monotone"), ("campaign_design", "broad"), ("campaign_design", "monotone")]
    for design, kind in settings:
        for seed in range(seeds):
            rng = np.random.default_rng(SEED+seed)
            if design == "campaign_design":
                x=real[FEATURES].to_numpy(float); ctrl=real.is_control.to_numpy(); batches=real.batch_id.to_numpy()
                background=np.zeros(len(FEATURES)); background[FEATURES.index("ethylene_glycol_M")]=1.
                # Known nuisance effects, correlated with ectoine in the actual design.
                nuisance=8*np.tanh(x[:,FEATURES.index("ethylene_glycol_M")]) + 1.0*x[:,FEATURES.index("fbs_pct")]
            else:
                d = np.tile(np.linspace(0, .15 if design == "one_sided" else .5, 11), 4)
                x=np.zeros((len(d)+4, len(FEATURES))); x[:len(d),E]=d
                ctrl=np.r_[np.zeros(len(d),bool),np.ones(4,bool)]
                batches=np.r_[np.repeat(np.arange(4),11),np.arange(4)].astype(str)
                background=np.zeros(len(FEATURES)); nuisance=np.zeros(len(x))
            offsets={b:rng.normal(0,8) for b in np.unique(batches)}
            y=np.where(ctrl,70.,true_response(x[:,E],kind)+nuisance)
            y=y+np.array([offsets[b] for b in batches])+rng.normal(0,10,len(y))
            model=BatchGP.fit(x,y,batches,ctrl,strategy="control")
            grid=np.linspace(0,.5,101)
            s,t=curve(model,background,"ectoin_M",grid,seed=SEED+seed)
            truth=true_response(grid,kind)
            if design=="campaign_design":truth=truth+8*np.tanh(1.)
            peak=float(grid[np.argmax(truth)])
            results.append(dict(design=design,truth=kind,seed=seed,true_peak=peak,
                peak_error=abs(s["peak_dose"]-peak), curve_rmse=float(np.sqrt(np.mean((t["mean"]-truth)**2))), **s))
            if seed==0: examples.append(t.assign(design=design,truth=kind,true_value=truth))
        print(f"  finished {design}/{kind}",flush=True)
    table=pd.DataFrame(results);table.to_csv(out / "synthetic_recovery.csv",index=False)
    pd.concat(examples).to_csv(out / "synthetic_example_curves.csv",index=False)
    summary=table.groupby(["design","truth"]).agg(runs=("seed","count"),median_peak_error=("peak_error","median"),
                   mean_curve_rmse=("curve_rmse","mean"),confident_interior_fraction=("probability_both_gains_over_delta",lambda s:float((s>=.95).mean())),
                   peak_within_0_05M_fraction=("peak_error",lambda s:float((s<=.05+1e-10).mean()))).reset_index()
    summary.to_csv(out / "synthetic_summary.csv",index=False)
    print(summary.to_string(index=False),flush=True)


def hypervolume(y):
    y=np.asarray(y); y=y[(y>0).all(axis=1)]
    # Union of rectangles from a zero reference point, for two maximized outputs.
    y=y[np.argsort(y[:,0])[::-1]]; area=0.; highest=0.
    for a,b in y:
        if b>highest:area+=a*(b-highest);highest=b
    return area


def objectives(d):
    return np.column_stack([true_response(d),.05+.18*np.exp(-.5*((d-.32)/.11)**2)])


def acquisition_experiment(out, seeds):
    import torch
    torch.set_num_threads(1)
    print(f"Sequential qLogNEHVI vs random, {seeds} seeds, eight added experiments; residual SD 10 pp, batch SD 8 pp",flush=True)
    grid=np.linspace(0,.5,101); oracle=hypervolume(objectives(grid)); records=[]
    for seed in range(seeds):
        for policy in ("qLogNEHVI","random"):
            rng=np.random.default_rng(SEED+seed)
            doses=[0.,.1,.4,.5]; batches=["0"]*4
            offsets={"0":rng.normal(0,8.)}; controls=[70+offsets["0"]+rng.normal(0,10.)]
            y=objectives(np.array(doses))+np.column_stack([offsets["0"]+rng.normal(0,10.,4),rng.normal(0,.003,4)])
            for step in range(9):
                truth=objectives(np.array(doses))
                records.append(dict(seed=seed,policy=policy,step=step,last_dose=doses[-1],
                    hypervolume_fraction=hypervolume(truth)/oracle,
                    best_true_viability=float(truth[:,0].max()), nearest_viability_peak=float(np.min(abs(np.array(doses)-.24)))))
                if step==8:break
                available=grid[np.min(abs(grid[:,None]-np.array(doses)[None,:]),axis=1)>.002]
                if policy=="random":chosen=float(rng.choice(available))
                else:
                    x=np.zeros((len(doses),len(FEATURES)));x[:,E]=doses
                    cx=np.zeros((len(controls),len(FEATURES)))
                    v=BatchGP.fit(np.vstack([x,cx]),np.r_[y[:,0],controls],np.r_[batches,[str(i) for i in range(len(controls))]],
                        np.r_[np.zeros(len(x),bool),np.ones(len(cx),bool)],strategy="control")
                    ms=_fit_regression(x,pd.Series(y[:,1]),y_noise=pd.Series(np.full(len(x),.003)))
                    m=snapshot_current(ms,x,y[:,1],np.array(batches))
                    bundle=SimpleNamespace(acquisition_bundle=torch_bundle([v,m]),gp_strategy_decision={"viability_strategy":"control"})
                    candidates=np.zeros((len(available),len(FEATURES)));candidates[:,E]=available
                    scores,_=shared_scores(bundle,x,candidates,(0.,0.))
                    chosen=float(available[np.argmax(scores)])
                b=str(step+1);offsets[b]=rng.normal(0,8.)
                new=objectives(np.array([chosen]))+np.array([[offsets[b]+rng.normal(0,10.),rng.normal(0,.003)]])
                y=np.vstack([y,new]);doses.append(chosen);batches.append(b);controls.append(70+offsets[b]+rng.normal(0,10.))
        print(f"  finished seed {seed+1}/{seeds}",flush=True)
    table=pd.DataFrame(records);table.to_csv(out / "sequential_acquisition.csv",index=False)
    summary=table.loc[table.step.eq(8)].groupby("policy").agg(runs=("seed","count"),median_hypervolume_fraction=("hypervolume_fraction","median"),
                   min_hypervolume_fraction=("hypervolume_fraction","min"),median_nearest_peak=("nearest_viability_peak","median"),
                   near_peak_fraction=("nearest_viability_peak",lambda s:float((s<=.05+1e-10).mean()))).reset_index()
    summary.to_csv(out / "sequential_summary.csv",index=False)
    print(summary.to_string(index=False),flush=True)


def figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.facecolor":"white"})
    curves=pd.read_csv(out / "ectoine_curves.csv"); contrasts=pd.read_csv(out / "ectoine_strategy_contrasts.csv")
    contexts=curves.context.unique()
    fig, axes=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
    names=["Fixed EG 1.06517 M (repeated recipe)", "Fixed EG 0.5 M", "Fixed EG 1 M + FBS 5%"]
    for ax,context,name in zip(axes,contexts,names):
        t=curves.loc[curves.context.eq(context)&curves.strategy.eq("control")]
        s=contrasts.loc[contrasts.context.eq(context)&contrasts.strategy.eq("control")].iloc[0]
        ax.fill_between(t.dose,t.lower,t.upper,alpha=.18,color="#176b9c",label="95% latent interval")
        ax.plot(t.dose,t["mean"],color="#176b9c",lw=2,label="Active GP")
        ax.axvline(s.peak_dose,color="#bc6132",ls=":")
        ax.set(title=name,xlabel="Ectoine (M)",ylabel="Predicted viability (%)",ylim=(-15,110))
        ax.text(.03,.04,f"Mean peak: {s.peak_dose:.3f} M\nP(both gains >5 pp): {s.probability_both_gains_over_delta:.1%}",transform=ax.transAxes)
    axes[0].legend(loc="upper left",fontsize=8)
    fig.suptitle("Ectoine: conditional predictions from the frozen Round 15 GP\nData through Round 14; other ingredients held fixed; no matched experimental titrations",fontsize=13)
    fig.savefig(out / "ectoine_posterior.png",dpi=190);plt.close(fig)
    ex=pd.read_csv(out / "synthetic_example_curves.csv")
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax,((design,truth),t) in zip(axes.ravel(),ex.groupby(["design","truth"],sort=False)):
        ax.fill_between(t.dose,t.lower,t.upper,color="#176b9c",alpha=.18)
        ax.plot(t.dose,t["mean"],color="#176b9c",label="Fitted GP")
        ax.plot(t.dose,t.true_value,"--",color="#bc6132",label="Known truth")
        ax.set(title=f"{design.replace('_',' ')} / {truth}",xlabel="Synthetic ectoine dose (M)",ylabel="Viability (%)",ylim=(-10,110))
    axes[0,0].legend();fig.suptitle("Synthetic stress tests — one prespecified seed shown; all seeds in CSV",fontsize=14)
    fig.savefig(out / "synthetic_checks.png",dpi=170);plt.close(fig)
    seq=pd.read_csv(out / "sequential_acquisition.csv")
    fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
    for policy,color in [("qLogNEHVI","#176b9c"),("random","#888888")]:
        g=seq.loc[seq.policy.eq(policy)].groupby("step").hypervolume_fraction
        med=g.median();ax.plot(med.index,med,color=color,label=policy)
        ax.fill_between(med.index,g.quantile(.1),g.quantile(.9),alpha=.15,color=color)
    ax.set(xlabel="Added synthetic experiments",ylabel="True hypervolume / grid oracle",ylim=(0,1.01),
           title="Can the production acquisition find interior tradeoffs?\nMedian and 10th–90th percentile across 12 seeds")
    ax.legend();fig.savefig(out / "acquisition_validation.png",dpi=180);plt.close(fig)


def verify(out):
    """Check numerical invariants, independent hypervolume, and input immutability."""
    import importlib.metadata
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    stats=json.loads((out / "actual_summary.json").read_text())
    for source, expected in stats["source_hashes"].items():
        assert sha(ROOT / source)==expected, f"Input changed during audit: {source}"
    synthetic=pd.read_csv(out / "synthetic_recovery.csv")
    sequential=pd.read_csv(out / "sequential_acquisition.csv")
    assert not synthetic.duplicated(["design","truth","seed"]).any()
    assert not sequential.duplicated(["policy","seed","step"]).any()
    assert sequential.hypervolume_fraction.between(0,1+1e-12).all()
    for name in ("ectoine_strategy_contrasts", "ectoine_all_observed_backgrounds", "ingredient_screen", "synthetic_recovery"):
        data=pd.read_csv(out / f"{name}.csv")
        assert data.probability_both_gains_over_delta.between(0,1).all()
        assert (data.probability_both_gains_over_delta<=data.probability_both_gains_positive+1e-12).all()
    y=objectives(np.linspace(0,.5,101))
    independent=Hypervolume(torch.tensor([0.,0.],dtype=torch.double)).compute(torch.tensor(y,dtype=torch.double))
    assert np.isclose(hypervolume(y),independent,rtol=1e-12)
    manifest=dict(seed=SEED,synthetic_runs=len(synthetic),sequential_runs=sequential.groupby(["policy","seed"]).ngroups,
                  synthetic_conditions=synthetic.groupby(["design","truth"]).size().to_dict(),
                  checks=["frozen model and training hashes", "no target/future outcomes in historical snapshots",
                          "joint posterior NumPy/BoTorch agreement", "independent BoTorch hypervolume agreement",
                          "source files unchanged during audit", "unique simulation keys and probability bounds"],
                  python=sys.version,versions={n:importlib.metadata.version(n) for n in ["numpy","pandas","scipy","scikit-learn","torch","botorch","gpytorch","matplotlib"]},
                  script_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!="validation_manifest.json"})
    manifest["synthetic_conditions"]={"/".join(k):v for k,v in manifest["synthetic_conditions"].items()}
    save_json(out / "validation_manifest.json",manifest)
    print("Verified source immutability, posterior adapter, output invariants and independent hypervolume.",flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument("--part",choices=["actual","synthetic","acquisition","figures","verify","all"],default="all")
    p.add_argument("--seeds",type=int,default=30);p.add_argument("--bo-seeds",type=int,default=12)
    p.add_argument("--output",type=Path,default=ROOT / "results/multi_objective_v2/reports/dose_response_20261022")
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.part in ("actual","all"):actual(a.output)
    if a.part in ("synthetic","all"):synthetic(a.output,a.seeds)
    if a.part in ("acquisition","all"):acquisition_experiment(a.output,a.bo_seeds)
    if a.part in ("figures","all"):figures(a.output)
    if a.part in ("verify","all"):verify(a.output)


if __name__=="__main__":main()
