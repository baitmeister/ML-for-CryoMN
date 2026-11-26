"""Render the decision records without recomputing campaign policy."""

from statistics import mean


V = "viability_percent"
F = "critical_axial_load_N_per_needle"
NAMES = {"pair_screened_hit": "Pair the screened hit", "repeat_frontier": "Repeat the frontier formulation",
         "optimizer primary": "Optimizer selection"}


def num(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def round_name(value):
    return f"R{int(value.split('_')[-1])}"


def brief(result):
    rounds, pending = result["rounds"], result["pending_plan"]
    lines = ["# CryoMN round decisions", "",
             "The supervisor reads each frozen proposal, records its priorities, then opens the validated results for that round. "
             "It starts with an empty decision history at the first full-mechanics round in this replay. The archived model retains its original training history.", ""]
    if rounds:
        start, last = (round_name(r["plan"]["batch_id"]) for r in (rounds[0], rounds[-1]))
        lines += [f"## What the project learned from {start} through {last}", ""]
        scored = [r["shortlist_score"] for r in rounds if r["shortlist_score"]["selected_mean"] is not None]
        if scored:
            selected = mean(s["selected_mean"] for s in scored)
            random = mean(s["random_mean"] for s in scored)
            hits, slots = sum(s["recovered"] for s in scored), sum(s["slots"] for s in scored)
            lines += [f"The frozen viability top three averaged **{selected:.2f}%**, compared with **{random:.2f}%** for three uniform picks "
                      f"from the same rankable nonreference slate ({selected - random:+.2f} percentage points). The shortlist recovered **{hits}/{slots}** observed top-three places. "
                      f"The comparison weights each of {len(scored)} completely observed rounds equally and uses the exact random expectation.", ""]
        lines += ["| Round | Intact / measured, all screens | Primary intact / measured | Viability top 3 / same-slate mean (%) | Viability ρ (n) | Force ρ (n) | Cumulative pairs | Frontier area |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rounds:
            s, v, f = r["shortlist_score"], r["viability"], r["force"]
            lines.append(f"| {round_name(r['plan']['batch_id'])} | {r['intact']}/{r['gate_measured']} | {r['primary_intact']}/{r['primary_gate_measured']} | "
                         f"{num(s['selected_mean'])} / {num(s['random_mean'])} | {num(v['rho'], 3)} ({v['n']}) | {num(f['rho'], 3)} ({f['n']}) | "
                         f"{r['cumulative_pairs']} | {num(r['hypervolume'])} |")
        lines += ["", "Each pair is an intact-passing, nonreference formulation with both endpoints measured in the same round. The frontier area is the area dominated by those observations above a fixed origin of 0% viability and 0 N/loaded needle. Its units are percentage points × N/needle. Historical compression measurements cover a selected subset.", "",
                  "| Frozen forecasts, pooled | n | Spearman ρ | Mean of within-round ρ | MAE | Nominal 95% interval hits | Mean width | Mean interval score |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for key, title in [("viability", "MSC viability (pp)"), ("force", "Terminal force (N/loaded needle)")]:
            p = result["totals"][key]
            within = [r[key]["rho"] for r in rounds if r[key]["rho"] is not None]
            digits = 2 if key == "viability" else 3
            lines += [f"| {title} | {p['n']} | {num(p['rho'], 3)} | {num(mean(within) if within else None, 3)} | {num(p['mae'], digits)} | "
                      f"{p['interval_hits']}/{p['interval_n']} | {num(p['interval_width'], digits)} | {num(p['interval_score'], digits)} |"]
        lines += ["", "Intervals use the archived proposal mean ± 1.96 standard deviations. The interval score equals width plus 40 times the distance outside the bounds. Lower scores are better. Within-round correlations receive equal weight. The shortlist ranks viability alone.", "",
                  "## Which scientific questions could the archive answer?", "",
                  "The policy first requests paired evidence for the highest-viability intact formulation lacking force in the latest round. It also identifies the highest-viability point on the cumulative frontier with paired evidence in only one replayed round. It reserves a repeat slot when all previous primary patches passed formation. A missing result stays open.", "",
                  "| Round revealed | Prior screened hit on this slate | Prior frontier target on this slate | First-choice mechanical nominations with pairs / failed gate / unknown |",
                  "|---|---|---|---:|"]
        for r in rounds:
            answers, cov = r["resolutions"], r["allocation_evidence"]
            statuses = [answers.get(k, {}).get("status", "initial round") for k in NAMES if k != "optimizer primary"]
            lines.append(f"| {round_name(r['plan']['batch_id'])} | {statuses[0]} | {statuses[1]} | {cov['paired']} / {cov['failed_gate']} / {cov['unresolved']} |")
        paired_slots, failed_slots, missing_slots = [sum(r["allocation_evidence"][k] for r in rounds) for k in ("paired", "failed_gate", "unresolved")]
        lines += ["", f"Of {paired_slots + failed_slots + missing_slots} first-choice mechanical nominations, the archive contains {paired_slots} matching paired results and {failed_slots} observed formation failures before any backup promotion. "
                  f"It leaves {missing_slots} unresolved. The other choices could have changed subsequent learning, so this replay cannot estimate the yield of an alternative campaign.", ""]
        closed = sum(q['answered_in'] is not None for q in result['question_ledger'])
        lines += [f"The question ledger contains {len(result['question_ledger'])} distinct target questions, including those raised at the latest close. "
                  f"Later matched measurements or formation failures resolved {closed}. Open requests are checked again when their exact formulation returns.", ""]
        for r in rounds:
            answer = r["resolutions"].get("pair_screened_hit", {})
            if answer.get("status", "").startswith("paired,"):
                target = r["plan"]["requests"]["pair_screened_hit"]["target"]
                point = answer["observed"]
                lines += [f"**{round_name(target['batch_id'])} → {round_name(point['batch_id'])}.** {target['recipe']} passed formation at {target[V]:.2f}% viability. "
                          f"The frozen next-round forecasts were {num(point['viability_forecast'])}% and {num(point['force_forecast'], 3)} N/needle. The researcher measured "
                          f"{num(point[V])}% and {num(point[F], 3)} N/needle. Individual force readings were "
                          + ", ".join(num(v, 3) for v in point[F + "_readouts"]) + f" N/needle. The decision record reads *{answer['status']}*.", ""]
        lines += ["## Measured trade-offs still on the frontier", "",
                  "| Round | Formulation at recorded concentrations | Viability (%) | Terminal force (N/loaded needle) |",
                  "|---|---|---:|---:|"]
        for p in result["frontier"]:
            lines.append(f"| {round_name(p['batch_id'])} | {p['recipe']} | {num(p[V])} | {num(p[F], 3)} |")
        lines += ["", "These are point estimates from selected batch means. Their order can change with independent preparations.", ""]
    if pending:
        lines += [f"## {round_name(pending['batch_id'])} briefing before results", "",
                  f"**Round priority** — {pending['focus'].capitalize()}. The archived slate has {pending['screens']} screens and {len(pending['primary_ids'])} primary mechanical slots, conditional on formation.", ""]
        if pending.get('agent_review'):
            review = pending['agent_review']
            decision = {"keep_default": "Keep the default allocation", "release_confirmation_slot": "Return the repeat slot to the optimizer"}[review['decision']]
            lines += [f"**Agent decision** — {decision} ({review['status']}).", ""]
            for label, key in [('Research question', 'question'), ('Competing explanation', 'alternative'), ('If supported', 'if_supported'), ('If refuted', 'if_refuted'), ('Critic', 'critique')]:
                if review.get(key):
                    lines += [f"**{label}** — {review[key]}", ""]
        for kind, req in pending["requests"].items():
            target = req["target"]
            lines += [f"**{NAMES[kind]}.** {req['question']} The target is {target['recipe']}. The prior round recorded {num(target[V])}% viability"
                      + (f" and {num(target[F], 3)} N/needle." if target[F] is not None else " and left force unmeasured.")
                      + f" Its position on the archived slate is *{req['frozen_role']}*. {req['decision_rule']}", ""]
            if req["matched_forecast"]:
                pred = req["matched_forecast"]
                lines += [f"The archived forecast for selection rank {pred['selection_rank']} is {num(pred['viability_forecast'])}% and {num(pred['force_forecast'], 3)} N/needle.", ""]
        draft = pending["new_project_draft"]
        lines += ["### How the new project would spend its mechanical budget", "",
                  "| Priority | Purpose | Formulation |", "|---:|---|---|"]
        for ix, slot in enumerate(draft["compression_priorities"], 1):
            purpose = NAMES.get(slot["purpose"], slot["purpose"])
            lines.append(f"| {ix} | {purpose} | {slot['recipe']} |")
        added = len(draft['added_screen_ids'])
        lines += ["", f"The provisional portfolio has {len(draft['screen_ids'])} distinct screens and {len(draft['ordered_backup_ids'])} ordered mechanical backups. "
                  f"It substitutes {added} {'screen' if added == 1 else 'screens'}. Bench outcomes for this briefing are pending at the chosen cutoff."]
        for displaced in draft['displaced_compression']:
            lines += ["", f"The cost of reserving a slot is the original mechanical primary at screening rank {displaced['selection_rank']} ({displaced['recipe']}). "
                      f"Its frozen forecasts were {num(displaced['viability_forecast'])}% and {num(displaced['force_forecast'], 3)} N/needle."]
    else:
        lines += ["## Next decision", "", f"**{result['next_agenda']['focus'].capitalize()}.** A frozen next proposal is needed to assign the available slots."]
    return "\n".join(lines).rstrip()


def plot(result, path):
    """Plot measured trade-offs and within-slate rankings, preserving both cohorts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds, points = result["rounds"], result["paired_points"]
    if not rounds or not points:
        raise ValueError("The comparison figure requires a completed round with paired evidence")
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.8))
    fig.subplots_adjust(left=.073, right=.978, bottom=.25, top=.75, wspace=.30)
    ink, teal, grey, orange = "#183748", "#087567", "#71818a", "#b64f2b"
    for ax in (left, right):
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c4cecf")
        ax.tick_params(colors=ink, labelsize=10, length=3)
        ax.grid(axis="y", color="#e0e6e7", lw=.8)
        ax.set_axisbelow(True)
    last = rounds[-1]["plan"]["batch_id"]
    for old, color, marker, label in [(True, grey, "o", "Earlier rounds"), (False, orange, "D", round_name(last))]:
        group = [p for p in points if (p["batch_id"] != last) == old]
        if group:
            left.scatter([p[V] for p in group], [p[F] for p in group], s=48, marker=marker,
                         c=color, alpha=.88, edgecolor="white", lw=.9, label=label, zorder=3)
    front = result["frontier"]
    left.scatter([p[V] for p in front], [p[F] for p in front], facecolors="none", edgecolors=ink,
                 linewidths=1.2, s=118, label="Observed frontier", zorder=4)
    left.set(xlabel="Measured post-thaw viability (%)", ylabel="Terminal force (N/loaded needle)", xlim=(0, 100), ylim=(0, max(p[F] for p in points)*1.12))
    left.set_title("Which trade-offs were measured?", loc="left", color=ink, weight="bold", fontsize=13, pad=13)
    left.legend(frameon=False, fontsize=9, loc="upper right")
    xs = [int(r["plan"]["batch_id"].split("_")[-1]) for r in rounds]
    for key, color, marker, label in [("selected_mean", teal, "o", "Top three by frozen viability forecast"),
                                      ("random_mean", grey, "s", "Uniform choice from the same slate")]:
        right.plot(xs, [r["shortlist_score"][key] for r in rounds], color=color, lw=2, marker=marker,
                   ms=5.5, label=label, linestyle="-" if key == "selected_mean" else "--")
    right.set(xlabel="Validation round", ylabel="Mean measured viability (%)", ylim=(0, 100), xticks=xs)
    right.set_title("What did the shortlist recover?", loc="left", color=ink, weight="bold", fontsize=13, pad=13)
    right.legend(frameon=False, fontsize=9, loc="lower right")
    fig.text(.073, .918, "What a round supervisor can learn from the archive", color=ink, fontsize=21, weight="bold")
    fig.text(.073, .861, f"CryoMN validation rounds {xs[0]}–{xs[-1]}   |   MSC post-thaw viability and whole-patch compression", color=grey, fontsize=11)
    fig.text(.073, .103, f"Each point is one intact, nonreference formulation–round mean with both endpoints measured (n = {len(points)}).", color=ink, fontsize=10)
    fig.text(.073, .064, "The shortlist comparison uses candidates on the same historical slate. Untested formulations have unknown outcomes.", color=ink, fontsize=10)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
