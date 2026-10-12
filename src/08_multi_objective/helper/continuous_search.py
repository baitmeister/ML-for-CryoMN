"""Concentration refinement of the approved, already fitted posterior.

Ingredient sets are fixed during each solve. Existing presence thresholds keep
the empirical exact-combination score continuous within that set. Every output
also passes the ordinary candidate validators; no GP is fitted here.
"""
from collections import Counter
import warnings

import numpy as np
import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.exceptions.warnings import OptimizationWarning

from .candidates import stable_formulation_id
from .config import nested_get
from .feasibility import ROUND5_POLICY_VERSION
from .gp_strategy import shared_acquisition
from .registry import presence_threshold


class ContinuousSearchError(RuntimeError):
    def __init__(self, message, metadata=None):
        super().__init__(message)
        self.metadata = metadata or {}


class MaskAcquisition(AcquisitionFunction):
    """Shared qLogNEHVI plus the exact existing intact/chemistry adjustment."""

    def __init__(self, acquisition, registry, config, policy, evidence, mask, lower, upper):
        super().__init__(model=acquisition.model)
        if not policy.active or policy.mechanics_mode != 'empirical_feasibility_weighted':
            raise ValueError('Shared continuous search requires the active empirical intact policy')
        self.acquisition = acquisition
        self.mask = tuple(mask)
        self.n_features = len(registry.feature_names)
        self.policy = policy
        features = [registry.feature_names[i] for i in mask]
        self.register_buffer('lower', torch.as_tensor(lower, dtype=torch.double))
        self.register_buffer('spread', torch.as_tensor(upper - lower, dtype=torch.double))
        self.register_buffer('embedding', torch.eye(self.n_features, dtype=torch.double)[list(mask)])
        ranges = [max(registry.get_by_feature(f).upper_bound - registry.get_by_feature(f).lower_bound, 1e-12) for f in features]
        same = evidence.loc[evidence.active_ingredient_set.map(lambda value: tuple(value) == tuple(features))]
        self.register_buffer('evidence_x', torch.as_tensor(same[features].to_numpy(float, copy=True), dtype=torch.double))
        self.register_buffer('evidence_pass', torch.as_tensor(same.intact_patch_formation_pass.to_numpy(float) >= .5, dtype=torch.double))
        self.register_buffer('ranges', torch.tensor(ranges, dtype=torch.double))
        molar = [registry.get_by_feature(f).unit == 'M' and registry.get_by_feature(f).penalize_single_over_500mM for f in features]
        self.register_buffer('molar', torch.tensor(molar, dtype=torch.double))
        self.count_penalty = float(nested_get(config, 'penalties.active_ingredient_excess_weight', .05)) * max(0, len(mask) - int(nested_get(config, 'penalties.active_ingredient_soft_limit', 8)))
        self.molar_limit = float(nested_get(config, 'penalties.single_molar_ingredient_limit_M', .5))
        self.molar_weight = float(nested_get(config, 'penalties.single_molar_excess_weight', .2))

    def physical(self, X):
        return (self.lower + X * self.spread) @ self.embedding

    def adjustment(self, X):
        if X.shape[-2] != 1:
            raise ValueError('Fixed-mask refinement uses q=1 sequential generation')
        active = (self.lower + X * self.spread).squeeze(-2)
        # vector_norm has a finite zero subgradient at exact evidence locations.
        distances = torch.linalg.vector_norm((active.unsqueeze(-2) - self.evidence_x) / self.ranges, dim=-1) / len(self.mask) ** .5
        weights = (1 - distances / self.policy.evidence_radius).clamp_min(0)
        passes = (weights * self.evidence_pass).sum(-1)
        probability = (self.policy.beta_prior_pass + passes) / (self.policy.beta_prior_pass + self.policy.beta_prior_fail + weights.sum(-1))
        chemistry = self.count_penalty + self.molar_weight * ((active - self.molar_limit).clamp_min(0) * self.molar).sum(-1)
        return probability.clamp_min(self.policy.numerical_probability_floor).log() - chemistry

    def forward(self, X):
        return self.acquisition(self.physical(X)) + self.adjustment(X)


def mixture_constraints(registry, config, mask, lower, upper, policy_version):
    """Translate existing linear chemistry limits into normalized coordinates.

    Other policies (preparation classifier, availability, similarity, support)
    retain their authoritative candidate/slate validation after optimization.
    """
    if policy_version != ROUND5_POLICY_VERSION:
        raise ValueError('Shared continuous search currently supports the Round 5 feasibility policy')
    ingredients = [registry.get_by_feature(registry.feature_names[i]) for i in mask]
    rules = nested_get(config, 'formulation_feasibility.round5_rules', {})
    grouped = lambda group: np.array([float(group in item.aggregate_groups) for item in ingredients])
    polymer = grouped('viscosity_active_polymer')
    protein = grouped('serum_protein')
    if polymer.sum() > int(rules.get('max_active_polymers', 1)):
        raise ValueError('Ingredient mask violates existing active-polymer limit')
    limits = [
        (polymer, rules.get('max_total_polymer_pct', 10.)),
        (protein, rules.get('max_total_serum_protein_pct', 10.)),
        (polymer + protein, rules.get('max_total_polymer_serum_pct', 15.)),
        (grouped('permeating_cpa'), rules.get('max_total_permeating_cpa_M', 2.5)),
        (grouped('sugar'), rules.get('max_total_sugar_M', .5)),
        (grouped('nonpermeating_osmolyte'), rules.get('max_total_nonpermeating_solute_M', .5)),
    ]
    if polymer.any():
        limits.extend([
            (grouped('sugar'), rules.get('max_sugar_M_with_polymer', .3)),
            (grouped('nonpermeating_osmolyte'), rules.get('max_nonpermeating_M_with_polymer', .3)),
        ])
    saturation = np.array([1 / item.aqueous_solubility_M if 'crystalline_solute' in item.aggregate_groups else 0. for item in ingredients])
    limits.append((saturation, rules.get('max_crystalline_solute_saturation_burden', .8)))
    return [(torch.arange(len(mask)), torch.as_tensor(-weights * (upper - lower), dtype=torch.double), float(weights @ lower - cap))
            for weights, cap in limits if weights.any()]


def select_refinement_masks(mask_scores, max_masks):
    """Reserve the best set per ingredient count, then fill by acquisition score.

    Rank each set by its best feasible starting recipe. If the budget cannot
    cover every count, choose the highest-scoring count representatives first.
    All ties and returned execution order are deterministic.
    """
    if max_masks < 1:
        raise ValueError('Refinement needs a positive ingredient-set budget')
    if any(not np.isfinite(score) for score in mask_scores.values()):
        raise ValueError('Ingredient-set ranking requires finite acquisition scores')
    ranked = sorted(mask_scores, key=lambda mask: (-mask_scores[mask], mask))
    champions = {}
    for mask in ranked:
        champions.setdefault(len(mask), mask)
    selected = list(champions.values())[:max_masks]
    selected.extend([mask for mask in ranked if mask not in selected])
    chosen = set(selected[:max_masks])
    return [mask for mask in ranked if mask in chosen]


def optimize_shared_candidates(*, models, train_x, candidate_pool, registry, config,
                               policy, evidence, masks, lower_bounds, upper_bounds,
                               reference, target, feasible, policy_version):
    """Use the existing 5-restart/150-iteration budget and up to 3*target solves."""
    metadata = dict(search_mode_requested='continuous_required', search_mode_actual='finite_pool_plus_continuous',
                    continuous_optimizer_enabled=True, continuous_optimizer_used=False,
                    continuous_optimizer_fallback=False, optimizer_attempt_count=0,
                    accepted_candidate_count=0, eligible_mask_count=len(masks), mask_records=[],
                    mask_ordering='best feasible seed acquisition score; reserve best set per ingredient count, then fill by score',
                    mask_budget=int(nested_get(config, 'continuous_qlognehvi.max_masks', 16)),
                    initialization='feasible finite-pool recipes', num_restarts_limit=5, maxiter=150,
                    pending_policy='sequential accepted continuous candidates; final ranking has no pending points',
                    score='shared_qlognehvi + log(empirical_intact_probability) - chemistry_penalty')
    accepted, pending, identities = [], [], set()
    rejections = Counter()
    pool_x = candidate_pool[registry.feature_names].to_numpy(float)
    floors = np.array([presence_threshold(f) for f in registry.feature_names])
    seed_sets = []
    for mask in masks:
        mask = tuple(mask)
        record = dict(features=[registry.feature_names[i] for i in mask], attempts=0, accepted=0)
        metadata.setdefault('mask_ranking_records', []).append(record)
        lo = np.maximum(lower_bounds[list(mask)], floors[list(mask)])
        hi = upper_bounds[list(mask)]
        seeds = pool_x[[tuple(np.flatnonzero(np.abs(row) >= floors)) == mask for row in pool_x]].copy()
        if len(seeds):
            seeds[:, [i for i in range(pool_x.shape[1]) if i not in mask]] = 0.
            seeds[:, list(mask)] = np.clip(seeds[:, list(mask)], lo, hi)
            seeds = np.asarray([row for row in seeds if feasible(row)], dtype=float)
        if not len(seeds) or np.any(hi <= lo):
            record['skip_reason'] = 'no feasible starting recipe within the fixed ingredient set'
            continue
        constraints = mixture_constraints(registry, config, mask, lo, hi, policy_version)
        seed_sets.append((mask, lo, hi, seeds, constraints, record))
    if not seed_sets:
        raise ContinuousSearchError('continuous_required: no feasible ingredient sets/starting recipes', metadata)
    try:
        with torch.random.fork_rng():
            torch.manual_seed(int(config.get('random_seed', 42)))
            # Use the exact same posterior, sampler and feasibility-adjusted
            # score as refinement, without pending candidates for this ranking.
            acquisition = shared_acquisition(models, train_x, reference)
            scores = {}
            for mask, lo, hi, seeds, _, record in seed_sets:
                wrapper = MaskAcquisition(acquisition, registry, config, policy, evidence, mask, lo, hi)
                starts = torch.tensor((seeds[:, list(mask)] - lo) / (hi - lo), dtype=torch.double).unsqueeze(-2)
                with torch.no_grad():
                    values = torch.cat([wrapper(chunk) for chunk in starts.split(64)])
                if not torch.isfinite(values).all():
                    raise ValueError('Nonfinite acquisition score while ranking ingredient sets')
                best = int(torch.argmax(values))
                scores[mask] = float(values[best])
                record.update(best_seed_score=scores[mask], feasible_seed_count=len(seeds),
                              best_seed_formulation_id=stable_formulation_id(dict(zip(registry.feature_names, seeds[best])), registry))
            # Each chosen set must receive a solve before any receives a second.
            # Limit the set budget to the candidate target so early success can
            # never exhaust the target before all selected sets are attempted.
            chosen = select_refinement_masks(scores, min(metadata['mask_budget'], target))
            by_mask = {item[0]: item for item in seed_sets}
            seed_sets = [by_mask[mask] for mask in chosen]
            champions = {}
            for rank, mask in enumerate(sorted(scores, key=lambda mask: (-scores[mask], mask)), 1):
                champions.setdefault(len(mask), mask)
                record = by_mask[mask][-1]
                record.update(acquisition_rank=rank, selected_for_refinement=mask in chosen,
                              best_for_ingredient_count=champions[len(mask)] == mask)
            metadata.update(masks_requested=len(chosen),
                            mask_records=[item[-1] for item in seed_sets],
                            mask_size_counts=dict(Counter(map(len, chosen))),
                            feasible_mask_size_counts=dict(Counter(map(len, scores))))
            for iteration in range(max(target * 3, len(seed_sets))):
                if len(accepted) >= target:
                    break
                mask, lo, hi, seeds, constraints, record = seed_sets[iteration % len(seed_sets)]
                acq = MaskAcquisition(shared_acquisition(models, train_x, reference, pending or None),
                                      registry, config, policy, evidence, mask, lo, hi)
                seed_tensor = torch.tensor((seeds[:, list(mask)] - lo) / (hi - lo), dtype=torch.double).unsqueeze(-2)
                with torch.no_grad():
                    values = torch.cat([acq(chunk) for chunk in seed_tensor.split(64)])
                    order = torch.argsort(values, descending=True, stable=True)[:5]
                starts = seed_tensor[order]
                metadata['optimizer_attempt_count'] += 1
                record['attempts'] += 1
                with warnings.catch_warnings(record=True) as emitted:
                    points, values = optimize_acqf(
                        acq_function=acq, bounds=torch.stack([torch.zeros(len(mask)), torch.ones(len(mask))]).double(),
                        q=1, num_restarts=len(starts), raw_samples=None, batch_initial_conditions=starts,
                        inequality_constraints=constraints or None, return_best_only=False,
                        options={'maxiter': 150, 'batch_limit': 5}, retry_on_optimization_warning=False,
                    )
                failures = [str(w.message) for w in emitted if issubclass(w.category, OptimizationWarning)]
                if failures:
                    raise RuntimeError('; '.join(failures))
                if not torch.isfinite(points).all() or not torch.isfinite(values).all():
                    raise RuntimeError('nonfinite optimizer output')
                for index in torch.argsort(values.reshape(-1), descending=True, stable=True).tolist():
                    vector = acq.physical(points[index]).detach().cpu().numpy().reshape(-1)
                    if not feasible(vector):
                        rejections['candidate_validation'] += 1
                        continue
                    identity = stable_formulation_id(dict(zip(registry.feature_names, vector)), registry)
                    if identity in identities:
                        rejections['duplicate_continuous_candidate'] += 1
                        continue
                    identities.add(identity)
                    accepted.append(vector)
                    pending.append(vector)
                    record['accepted'] += 1
                    break
    except Exception as exc:
        metadata.update(optimizer_error=f'{type(exc).__name__}: {exc}', accepted_candidate_count=len(accepted))
        raise ContinuousSearchError('continuous_required stopped: ' + metadata['optimizer_error'], metadata) from exc
    metadata.update(accepted_candidate_count=len(accepted), optimized_mask_count=sum(r['attempts'] > 0 for r in metadata['mask_records']),
                    optimized_mask_size_counts=dict(Counter(len(r['features']) for r in metadata['mask_records'] if r['attempts'] > 0)),
                    rejection_counts=dict(rejections), continuous_optimizer_used=bool(accepted),
                    target_candidate_count=target, target_reached=len(accepted) >= target,
                    continuous_optimizer_reason='shared posterior refinement completed')
    if not accepted:
        raise ContinuousSearchError('continuous_required: no valid continuous candidate', metadata)
    return np.asarray(accepted), metadata
