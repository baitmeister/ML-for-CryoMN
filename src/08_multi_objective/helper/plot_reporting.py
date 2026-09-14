"""Production figure bundles and explicit, reusable rendering evidence."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import pandas as pd
from . import campaign_plots as plots
from .plot_data import timeline_table, trust_table, decision_table
from .evaluation_metrics import (
    build_feasible_paired_objectives, _build_model_evaluation_frames, _paired_frame,
    _round_metrics, _cross_validated_predictions, endpoint_r2_history,
    summarize_prospective_metrics,
)
from .registry import load_registry

ENDPOINTS=['viability_percent','critical_axial_load_N_per_needle','intact_patch_formation_pass']


def write_plot(fig, table, directory, stem, context='', sources=None):
    """Preserve the rendering inputs alongside each PNG for later safe restyling."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    path=plots.save_png(fig,directory/(stem+'.png'))
    table_path=directory/(stem+'_source.csv')
    table.to_csv(table_path,index=False)
    metadata={'style_version':'selected_v1','context':context,'format':'transparent PNG','dpi':300,
              'source_sha256':hashlib.sha256(table_path.read_bytes()).hexdigest(),
              'sources':sources or [],'synthetic_data':False}
    meta=directory/(stem+'_metadata.json');meta.write_text(json.dumps(metadata,indent=2)+'\n')
    return [path,table_path,meta]


def write_decision(candidates,directory,prefix=''):
    prefix = prefix.rstrip('_')+'_' if prefix else ''
    if candidates.empty:return []
    table=decision_table(candidates)
    return write_plot(plots.decision_figure(table),table,directory,prefix+'candidate_decisions','Frozen proposal; stored assignments')


def write_pareto(formulations,observations,candidates,directory,prefix='',context='Current evidence'):
    prefix = prefix.rstrip('_')+'_' if prefix else ''
    feasible,excluded=build_feasible_paired_objectives(formulations,observations)
    table=pd.concat([feasible.assign(evidence_kind='feasible_observed'),excluded.assign(evidence_kind='excluded_observed'),candidates.assign(evidence_kind='proposal_prediction')],ignore_index=True)
    definition = feasible.get('mechanical_definition_id',pd.Series(dtype=str)).dropna().unique()
    fig = plots.pareto_figure(feasible,excluded,candidates)
    if len(definition):
        context += '; mechanical definition: ' + ', '.join(definition)
        fig.text(.5,.005,', '.join(definition),ha='center',fontsize=7)
        if list(definition) == ['terminal_force_08mm_after_1N_v1']:
            for ax in fig.axes:
                if 'load' in ax.get_ylabel().lower():
                    ax.set_ylabel('Terminal compression resistance\n(nominal N per loaded needle)')
    return write_plot(fig,table,directory,prefix+'observed_tradeoff',context)


def prepare_diagnostics(formulations,observations,frames=None):
    registry=load_registry()
    frames=frames if frames is not None else _build_model_evaluation_frames(formulations,observations,registry)
    paired=_paired_frame(formulations,observations,registry)
    metrics=_round_metrics(paired)
    paired_frames={}
    if not paired.empty:
        keys=set(zip(paired.formulation_id.astype(str),paired.batch_id.astype(str)))
        selected=observations.loc[[(str(row.formulation_id),str(row.batch_id)) in keys for row in observations.itertuples()]]
        for endpoint in ENDPOINTS[:2]:paired_frames[endpoint]=_cross_validated_predictions(formulations,selected,registry,endpoint)
    r2=endpoint_r2_history(formulations,observations,metrics,registry)
    return frames,paired_frames,metrics,r2


def diagnostic_source(frames,paired_frames,metrics,r2):
    tables=[frame.assign(panel='all_cv',endpoint=endpoint) for endpoint,frame in frames.items()]
    tables += [frame.assign(panel='paired_cv',endpoint=endpoint) for endpoint,frame in paired_frames.items()]
    tables += [metrics.assign(panel='progress'),r2.assign(panel='r2')]
    result=pd.concat(tables,ignore_index=True)
    return result.reindex(columns=list(dict.fromkeys([*result.columns,'panel','endpoint','actual','predicted','batch_id','viability_r2','load_r2','normalized_hypervolume','igd'])))


def diagnostic_inputs(table):
    frames={endpoint:table.loc[table.panel.eq('all_cv') & table.endpoint.eq(endpoint)] for endpoint in ENDPOINTS}
    paired={endpoint:table.loc[table.panel.eq('paired_cv') & table.endpoint.eq(endpoint)] for endpoint in ENDPOINTS[:2]}
    return frames,paired,table.loc[table.panel.eq('progress')],table.loc[table.panel.eq('r2')]


def write_diagnostics(inputs,directory,prefix='',context='Current evidence'):
    prefix = prefix.rstrip('_')+'_' if prefix else ''
    source=diagnostic_source(*inputs)
    figures=plots.diagnostic_figures(*inputs,context=context)
    generated=[]
    for index,fig in enumerate(figures,1):
        generated+=write_plot(fig,source,directory,prefix+f'diagnostics_{index}',context)
    return generated


def write_prospective(observations,table,metrics,candidates,directory,context,
                      include_publication_summary=False,formulations=None):
    timeline=timeline_table(observations,table,metrics,candidates)
    trust=trust_table(table)
    # A per-round metrics archive has no pooled row. Summarize only its frozen cohort.
    display_metrics=metrics
    if 'pooled_formal' not in metrics.get('scope',pd.Series(dtype=str)).values:
        display_metrics=summarize_prospective_metrics(table)
    mechanics=int(observations.endpoint.eq('critical_axial_load_N_per_needle').sum())
    figures=[('campaign_timeline',plots.timeline_figure(timeline),timeline),
             ('surrogate_trust',plots.trust_figure(trust,display_metrics,mechanics),trust)]
    if include_publication_summary:
        feasible,_=build_feasible_paired_objectives(formulations if formulations is not None else pd.DataFrame(),observations)
        source=pd.concat([timeline.assign(section='timeline'),trust.assign(section='trust'),feasible.assign(section='feasible')],ignore_index=True)
        figures.append(('publication_summary',plots.publication_figure(timeline,trust,display_metrics,feasible),source))
    generated=[]
    for stem,fig,source in figures:
        fig.text(.5,.008,context,ha='center',fontsize=8,color=plots.GRAY)
        generated+=write_plot(fig,source,directory,stem,context)
    return generated
