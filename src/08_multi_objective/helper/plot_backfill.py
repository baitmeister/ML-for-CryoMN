"""Rendering-only migration: use archived evidence, never refit historical models."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import pandas as pd
from PIL import Image
from .plot_reporting import (read_strategy_decisions, write_decision, write_pareto, write_diagnostics,
                             write_prospective, diagnostic_inputs, ENDPOINTS)

REPLACED = {
 'candidate_decisions': ['next_round_candidate_screen.png'],
 'observed_tradeoff': ['observed_performance_landscape.png'],
 'diagnostics_1': ['model_evaluation_overview.png','endpoint_r2_vs_round.png'],
 'diagnostics_2': ['multiobjective_paired_parity.png','normalized_hypervolume_igd_vs_round.png'],
 'campaign_timeline': ['prospective_error_by_round.png'],
 'surrogate_trust': ['prospective_prediction_vs_observed.png','prospective_gate_calibration.png'],
}


def archived_observations(table):
    """Use the archived formulation-round aggregate, not today's canonical measurements."""
    cols=['formulation_id','round_id','endpoint','observed_mean','mechanical_definition_id']
    frame=table.reindex(columns=cols).dropna(subset=['observed_mean']).drop_duplicates()
    if frame.duplicated(['formulation_id','round_id','endpoint']).any():
        raise ValueError('Conflicting archived observation aggregates')
    return frame.rename(columns={'round_id':'batch_id','observed_mean':'value'}).assign(source_type='archived_prospective')


def _promote(stage,destination,generated):
    """Validate every export before removing any superseded artifact."""
    for path in generated:
        if path.suffix=='.png':
            with Image.open(path) as im:
                if im.mode!='RGBA' or im.getchannel('A').getextrema()[0]!=0:
                    raise ValueError(f'PNG is not transparent: {path}')
                if abs(im.info.get('dpi',(0,0))[0]-300)>1:
                    raise ValueError(f'PNG is not 300 dpi: {path}')
    destination.mkdir(parents=True,exist_ok=True)
    promoted=[]
    for path in generated:
        target=destination/path.relative_to(stage)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);promoted.append(target)
    removed=[]
    for path in promoted:
        if path.suffix!='.png':continue
        for name in REPLACED.get(path.stem,[]):
            previous=path.parent/name
            if previous.exists():previous.unlink();removed.append(previous)
    # Counts and progression are retired only when their replacement summaries exist.
    stems={p.stem for p in promoted if p.suffix=='.png'}
    for stem,name in [('observed_tradeoff','endpoint_observation_counts.png'),('diagnostics_2','pareto_front_progression.png')]:
        if stem in stems:
            previous=destination/name
            if previous.exists():previous.unlink();removed.append(previous)
    return promoted,removed


def restyle_existing(results_root,include_publication_summary=False,round_id=None):
    root=Path(results_root)
    decisions=read_strategy_decisions(root)
    records=[]
    def render(destination,render_fn,sources):
        try:
            with tempfile.TemporaryDirectory(prefix='cryo_plot_') as name:
                stage=Path(name)
                generated=render_fn(stage)
                paths,removed=_promote(stage,destination,generated)
            records.append({'destination':str(destination.relative_to(root)),'status':'refreshed',
                'sources':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'generated':[str(p.relative_to(root)) for p in paths],
                'removed':[str(p.relative_to(root)) for p in removed]})
        except (ValueError,KeyError,TypeError) as exc:
            records.append({'destination':str(destination.relative_to(root)),'status':'skipped','reason':str(exc)})
    rounds=sorted((root/'rounds').glob('ROUND_*'))
    if round_id:rounds=[p for p in rounds if p.name==round_id]
    for directory in rounds:
        proposal=directory/'proposal/proposal.csv'
        if proposal.exists():
            candidates=pd.read_csv(proposal)
            required={'selection_rank','mechanical_test_recommended','mechanical_selection_rank','empirical_combination_pass_probability'}
            if required.issubset(candidates.columns):
                render(directory/'proposal/plots',lambda dest:write_decision(candidates,dest),[proposal])
            else:records.append({'destination':str((directory/'proposal/plots').relative_to(root)),'status':'skipped','reason':'Frozen proposal lacks active empirical probability or mechanical assignments; old plot preserved.'})
        report=directory/'reports'
        source=report/'tables/prospective_evaluation_table.csv';metric=report/'tables/prospective_metrics.csv'
        if source.exists() and metric.exists():
            table=pd.read_csv(source);metrics=pd.read_csv(metric)
            if set(table.round_id.dropna())!={directory.name}:
                records.append({'destination':str(report.relative_to(root)),'status':'skipped','reason':'Prospective archive does not contain exactly its declared round.'})
                continue
            observed=archived_observations(table)
            render(report/'plots',lambda dest:write_prospective(observed,table,metrics,pd.DataFrame(),dest,directory.name+' only; archived frozen evidence',strategy_decisions=decisions),[source,metric])
            # Trade-off is explicitly single-round evidence; no inference about older/literature data.
            render(report/'plots',lambda dest:write_pareto(pd.DataFrame(),observed,pd.DataFrame(),dest,context=directory.name+' only; archived observations'),[source])
        else:records.append({'destination':str(report.relative_to(root)),'status':'skipped','reason':'No stored prospective table and metrics; historical reports preserved.'})
        diagnostic=report/'plots/diagnostics_1_source.csv'
        cv=report/'tables/model_evaluation_table.csv'
        if diagnostic.exists():
            data=pd.read_csv(diagnostic)
            has_second=(report/'plots/diagnostics_2_source.csv').exists()
            render(report/'plots',lambda dest:write_diagnostics(diagnostic_inputs(data),dest,context=directory.name+' archived evaluations')[:6 if has_second else 3],[diagnostic])
            if not has_second:
                records.append({'destination':str((report/'plots/diagnostics_2.png').relative_to(root)),'status':'skipped','reason':'No archived paired CV/history tables; no historical models refitted.'})
        elif cv.exists():
            data=pd.read_csv(cv)
            frames={e:data.loc[data.endpoint.eq(e)] for e in ENDPOINTS}
            # No mechanics CV is not proof of no paired evidence. Keep unavailable panels explicit;
            # preserve separate old paired/R²/HV files because their source cannot be recovered.
            inputs=(frames,{},pd.DataFrame(),pd.DataFrame())
            def overview(dest):
                from .campaign_plots import diagnostic_figures
                from .plot_reporting import write_plot,diagnostic_source
                import matplotlib.pyplot as plt
                one,two=diagnostic_figures(*inputs,context=directory.name+' archived CV; paired history unavailable')
                plt.close(two)
                return write_plot(one,diagnostic_source(*inputs),dest,'diagnostics_1','Archived all-data CV; historical R² unavailable')
            # Only the overview is replaced, not an unrecoverable old R² figure.
            old_r2=report/'plots/endpoint_r2_vs_round.png'
            if old_r2.exists():
                records.append({'destination':str(report.relative_to(root)),'status':'skipped','reason':'Stored R² trend unavailable; existing diagnostics retained.'})
            else:render(report/'plots',overview,[cv])
            records.append({'destination':str((report/'plots/diagnostics_2.png').relative_to(root)),'status':'skipped','reason':'No archived paired CV/history tables; no historical models refitted.'})
    if not round_id:
        report=root/'reports/prospective';source=report/'tables/prospective_evaluation_table.csv';metric=report/'tables/prospective_metrics.csv'
        if source.exists() and metric.exists():
            table=pd.read_csv(source);metrics=pd.read_csv(metric);observed=archived_observations(table)
            proposal=root/'next_round/next_round_candidates.csv'
            candidates=pd.read_csv(proposal) if proposal.exists() else pd.DataFrame()
            sources=[source,metric]+([proposal] if proposal.exists() else [])
            refresh_publication = include_publication_summary or (report/'plots/publication_summary.png').is_file()
            render(report/'plots',lambda dest:write_prospective(observed,table,metrics,candidates,dest,'Cumulative campaign; archived frozen evidence',refresh_publication,strategy_decisions=decisions),sources)
            render(root/'reports/plots',lambda dest:write_pareto(pd.DataFrame(),observed,candidates,dest,context='Cumulative campaign; archived observations and current proposal'),sources)
    # New index, rather than rewriting scientific summaries or their historical provenance.
    for directory in {root/record['destination'] for record in records if record['status']=='refreshed'}:
        images=sorted(directory.glob('*.png'))
        (directory/'README.md').write_text('# Production plots\n\nTransparent PNGs, 300 dpi. Unrefreshed historical figures may retain their original style.\n\n'+''.join(f'- [{p.stem}]({p.name})\n' for p in images))
    manifest=root/'reports/plot_backfill_manifest.json';manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({'policy':'render archived evidence only; never refit historical models','records':records},indent=2)+'\n')
    return manifest
