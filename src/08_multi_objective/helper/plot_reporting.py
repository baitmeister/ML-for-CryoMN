"""Production figure bundles and explicit, reusable rendering evidence."""
from __future__ import annotations
import json
import hashlib
import re
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
    metadata_path=Path(directory).parent/'selection_metadata.json'
    decision={}
    if metadata_path.exists():decision=json.loads(metadata_path.read_text()).get('gp_strategy_decision',{})
    from .batch_gp import NAMES
    name=NAMES.get(decision.get('viability_strategy','current'),'Current GP')
    table['production_gp']=name
    fig=plots.decision_figure(table)
    fig.text(.09,.012,name+' | '+decision.get('decision_id','historical production'),fontsize=8)
    return write_plot(fig,table,directory,prefix+'candidate_decisions','Frozen proposal; stored assignments; '+name)


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
                      include_publication_summary=False,formulations=None,strategy_decisions=None):
    timeline=timeline_table(observations,table,metrics,candidates)
    decisions=strategy_decisions or {}
    timeline['production_strategy']=timeline.round_id.map(lambda r: decisions.get(r, {}).get('viability_strategy', 'current'))
    timeline['strategy_decision_id']=timeline.round_id.map(lambda r: decisions.get(r, {}).get('decision_id', 'historical_current'))
    trust=trust_table(table)
    trust['production_strategy']=trust.round_id.map(lambda r: decisions.get(r, {}).get('viability_strategy', 'current'))
    trust['interval_type']='stored official prediction interval; not reconstructed future-batch uncertainty'
    if decisions:
        context += ' | Recorded GP decisions: ' + ', '.join(f"{r}: {d['decision_id']}" for r,d in sorted(decisions.items()))
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


def mechanical_report_table(observations, candidates=None, batch_id=None):
    """Build replicate-level terminal-force evidence with visible QC summaries."""
    status = observations.loc[
        observations.get('endpoint', pd.Series('', index=observations.index)).eq('terminal_force_08mm_status')
    ].copy()
    if batch_id is not None and 'batch_id' in status:
        status = status.loc[status.batch_id.astype(str).eq(str(batch_id))]
    if status.empty or 'analysis_provenance' not in status:
        return pd.DataFrame()
    rank_by_formulation = {}
    if candidates is not None and not candidates.empty:
        ranks = candidates[['formulation_id', 'selection_rank']].drop_duplicates('formulation_id')
        rank_by_formulation = dict(zip(ranks.formulation_id.astype(str), ranks.selection_rank))
    rows = []
    for row in status.itertuples():
        if not isinstance(row.analysis_provenance, str) or not row.analysis_provenance.strip():
            continue
        analysis = json.loads(row.analysis_provenance)
        if analysis.get('status') != 'complete':
            continue
        notes = str(getattr(row, 'notes', '') or '')
        rows.append({
            'formulation_id': str(row.formulation_id),
            'formulation_number': rank_by_formulation.get(str(row.formulation_id), str(row.formulation_id)),
            'batch_id': str(row.batch_id),
            'replicate_id': str(row.replicate_id),
            'terminal_force_N_total': analysis['endpoint_N_total'],
            'terminal_force_N_per_needle': analysis['endpoint_N_per_needle'],
            'apparent_secant_stiffness_N_per_mm_total': analysis['apparent_secant_stiffness_N_per_mm_total'],
            'apparent_secant_stiffness_N_per_mm_per_needle': analysis['apparent_secant_stiffness_N_per_mm_per_needle'],
            'loaded_needle_count': analysis['loaded_needle_count'],
            'trigger_force_N': analysis['trigger_force_N'],
            'trigger_displacement_mm': analysis['contact_displacement_mm'],
            'terminal_displacement_mm': analysis['terminal_displacement_mm'],
            'mechanical_definition_id': analysis['definition_id'],
            'stiffness_definition_id': analysis['stiffness_definition_id'],
            'stiffness_formula': analysis['stiffness_formula'],
            'stiffness_qc': analysis.get('stiffness_qc', 'not_recorded'),
            'source_file': analysis['source_file'],
            'source_file_hash': analysis['source_file_hash'],
            'replicate_lost': 'lost' in notes.lower(),
            'notes': notes,
            'analysis_provenance': row.analysis_provenance,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table['_sort'] = pd.to_numeric(table.formulation_number, errors='coerce')
    table = table.sort_values(['_sort', 'replicate_id'], kind='mergesort').drop(columns='_sort')
    for column, prefix in [
        ('terminal_force_N_per_needle', 'terminal_force'),
        ('apparent_secant_stiffness_N_per_mm_per_needle', 'apparent_secant_stiffness'),
    ]:
        grouped = table.groupby('formulation_id')[column]
        table[prefix + '_replicate_count'] = grouped.transform('count')
        table[prefix + '_mean'] = grouped.transform('mean')
        table[prefix + '_sample_sd'] = grouped.transform('std')
        table[prefix + '_min'] = grouped.transform('min')
        table[prefix + '_max'] = grouped.transform('max')
    return table


def write_mechanical(observations, candidates, directory, batch_id=None, context='Completed round'):
    """Write trace-level QC and formulation-level terminal-method figures."""
    table = mechanical_report_table(observations, candidates, batch_id)
    if table.empty:
        return []
    from .instron import _read_bluehill_csv
    from .terminal_force import numeric_curve
    generated = []
    directory = Path(directory)
    # Verify and parse every source before overwriting any report artifact.
    traces = []
    for row in table.itertuples():
        analysis = json.loads(row.analysis_provenance)
        source = Path(analysis['source_file'])
        frame = _read_bluehill_csv(source, expected_sha256=analysis['source_file_hash'])
        columns = [analysis['force_column'], analysis['displacement_column'], analysis['time_column']]
        arrays, _ = numeric_curve(frame, columns, analysis['settings'])
        traces.append((row, analysis, arrays))
    for row, analysis, arrays in traces:
        force, displacement, time = arrays
        terminal_end = int(analysis['terminal_bracket_indices'][1])
        indexes = pd.Series(range(len(force)))
        trace = pd.DataFrame({
            'sample_index': indexes,
            'time_s': time,
            'displacement_mm': displacement,
            'force_N': force,
            'analysis_window': indexes.between(int(analysis['trigger_index']), terminal_end),
            'formulation_number': row.formulation_number,
            'replicate_id': row.replicate_id,
            'source_file_hash': analysis['source_file_hash'],
        })
        token = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(row.replicate_id)).strip('_')
        stem = f'mechanical_trace_formulation_{row.formulation_number}_{token}'
        fig = plots.mechanical_trace_figure(trace, analysis, row.formulation_number, row.replicate_id)
        generated += write_plot(
            fig, trace, directory, stem,
            context=f'{context}; {analysis["definition_id"]}',
            sources=[{'path': analysis['source_file'], 'sha256': analysis['source_file_hash']}],
        )
    summary_source = table.drop(columns='analysis_provenance')
    generated += write_plot(
        plots.mechanical_summary_figure(summary_source, context),
        summary_source,
        directory,
        'mechanical_summary',
        context=f'{context}; terminal force and apparent secant stiffness',
        sources=[
            {'path': row.source_file, 'sha256': row.source_file_hash}
            for row in summary_source.itertuples()
        ],
    )
    return generated
