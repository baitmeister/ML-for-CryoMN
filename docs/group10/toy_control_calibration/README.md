# Synthetic demonstration of viability control calibration

These are invented data, not Group 10 outcomes, forecasts or evidence for choosing a strategy. No campaign records or production settings were changed. Run `generate.py` to reproduce all figures (seed 42). Source CSVs and metadata accompany each transparent 300-dpi PNG.

## Design

Four fictional formulations A–D have underlying viability 45%, 60%, 72% and 65%. A fictional viability control has baseline 65%: this is an assumption for teaching, not a claim about the DMSO–sucrose recipe. Six preceding synthetic batches have shared shifts of −12, +8, −5, +12, −8 and +5 percentage points. Each has all four formulations plus one control, with small measurement noise. Repeating all four formulations across batches makes this deliberately favorable and more identifiable than the actual campaign; it is not a recommendation for more experimental slots.

The script fits the actual BatchGP implementation twice: Proposed GP — campaign-only + batch effect (controls excluded), and Proposed GP — campaign-only + batch effect + control calibration (control baseline separate). The latter is then conditioned on one new control result of 47%, without refitting parameters or revealing ordinary outcomes.

## Read the figures

1. [Training history](01_shared_batch_history.png): formulations and control move together.
2. [Before and after revealing the control](02_control_conditioning.png): the new batch's true shift is −18 points. Before observing its reference, historical controls cannot tell us whether this particular new batch is low. After observing 47%, the control-calibrated model lowers its batch-measurement predictions and narrows their intervals. Mean absolute error versus the hidden toy batch means decreases from 18.55 to 1.01 points. This is an illustrative result, not an estimate of real-world performance. Intervals are future-observation intervals with fitted parameters treated as fixed.
3. [Failure case](03_when_calibration_fails.png): keep the exact same control result, but let its drop be a control-specific error while ordinary formulations have no batch shift. The same correction now worsens error from 0.55 to 16.99 points. The model cannot distinguish these causes from one control measurement alone.
4. [Mechanical independence](04_mechanics_unchanged.png): a separate synthetic mechanical GP's posterior stays exactly identical when only viability is conditioned. This illustrates the independent-output production architecture, not a fit to actual force data.

The hidden values plotted as crosses are synthetic noise-free batch means, not held-out noisy samples. No synthetic coverage claim is made. The example omits sparse-history and formulation-by-batch interactions; these would weaken the simple correction. Raw outcomes are never rewritten.

## What the control model assumes

Ordinary viability: y(i,b) = f(i) + u(b) + residual.
Control viability: c(b) = reference_baseline + u(b) + control_residual.

The shared u(b) is a viability batch effect in percentage points. It is not measured cell health. The current implementation fits a common residual-noise level rather than a separately estimated control-specific noise level. The reference baseline is inferred with uncertainty; calibration is not a direct subtraction of 65 minus the control result. Sparse references leave baseline and shift uncertain. Two historical control batches satisfy a software eligibility check, not a scientific guarantee of reliability.

The plotted predictions concern this batch's measurements. Acquisition instead uses the underlying formulation-response posterior without adding a fresh batch shift or measurement noise. Conditioning can also update correlated latent estimates, but it does not redefine the optimization objective as this batch's raw viability.

## Mechanics

The production strategy installer replaces the viability model and preserves the Current GP mechanical posterior, trained on comparable terminal-force labels. Its independent-output acquisition bundle does not transfer a viability batch correction to force. Valid mechanical results for the reference recipe remain ordinary mechanical evidence, with no normalization or reserved control slot.

A mechanical batch effect, if later justified by fabrication or instrument variation, would be a separate endpoint-specific model in force units. It is not currently added and should not be inferred from viability-control measurements. qLogNEHVI can change candidate preferences when its viability model changes even though every mechanical prediction remains the same.
