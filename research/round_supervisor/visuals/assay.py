"""Connect the joint objective to an observed pair of patch compression traces."""

from hashlib import sha256
from pathlib import Path

from matplotlib import patches
import numpy as np
import pandas as pd

from .evidence import F, ROOT
from .layout import BLUE, GREY, MINT, ORANGE, PALE, PAPER, PEACH, SKY, TEAL, Plate, arrow, ax, canvas, node


def assay_schematic(fig, data):
    a = canvas(fig)
    a.text(.8,94,'Whole-patch compression',fontsize=15,weight='bold',va='top')
    a.text(4,80,'Applied load',fontsize=11.2,color=BLUE)
    arrow(a,[(24,87),(24,65)],color=BLUE)
    a.add_patch(patches.Rectangle((6,58),36,6,fc=SKY,ec=BLUE,lw=1.25))
    a.text(24,61,'Compression platen',ha='center',va='center',color=BLUE,fontsize=10.4)
    for center in range(10,41,5):
        a.add_patch(patches.Polygon([(center-2.1,27),(center,48),(center+2.1,27)],closed=True,fc=MINT,ec=TEAL,lw=1.1))
    a.add_patch(patches.Rectangle((6,21),36,6,fc=TEAL,ec=TEAL))
    a.add_patch(patches.Rectangle((3,17),42,3.7,fc=PALE,ec=GREY,lw=.8))
    a.text(24,8,'Cryomicroneedle patch, illustrative side view',ha='center',fontsize=10.6,color=GREY)

    node(a,53,72,44,24,'One formulation in one validation round','Match recipe identity, round and formation result.')
    node(a,50.5,36,22.5,25,'Viability','Average post-thaw\ncell readouts (%).',color=BLUE,face=SKY,title_size=12,body_size=10.8)
    node(a,76.5,36,22.5,25,'Mechanical force','Average terminal force\nper loaded needle.',color=ORANGE,face=PEACH,title_size=12,body_size=10.8)
    node(a,53,0.8,44,24,'One observed point on the plot','An intact pass and both endpoint means are required.')
    for x in (61.75,87.75):
        arrow(a,[(75,72),(75,66),(x,66),(x,61)])
        arrow(a,[(x,36),(x,30),(75,30),(75,25)])


def leader_traces(data):
    directory = ROOT/'results/multi_objective_v2/rounds/ROUND_019/reports/plots'
    raw = data['raw']
    observed = raw.loc[raw.batch_id.eq('ROUND_019') & raw.formulation_id.eq(data['leader']) & raw.endpoint.eq(F)].set_index('replicate_id')
    cases = pd.read_csv(directory/'mechanical_summary_source.csv')
    selected = cases[cases.formulation_id.eq(data['leader'])].sort_values('replicate_id')
    if len(selected)!=2 or len(observed)!=2:
        raise ValueError('Expected two archived force readouts for the R19 follow-up')
    result = []
    for r in selected.itertuples():
        table = pd.read_csv(directory/f'mechanical_trace_formulation_{int(r.formulation_number)}_{r.replicate_id}_source.csv')
        archive = ROOT/'data/raw/instron/ROUND_019'/Path(r.source_file).name
        if (not table.source_file_hash.eq(r.source_file_hash).all() or sha256(archive.read_bytes()).hexdigest()!=r.source_file_hash
                or observed.at[r.replicate_id,'raw_file_hash']!=r.source_file_hash):
            raise ValueError(f'Trace identity mismatch in {r.replicate_id}')
        window = table.loc[table.analysis_window]
        x = window.displacement_mm.to_numpy()-r.trigger_displacement_mm
        y = window.force_N.to_numpy()
        terminal = np.interp(.8,x,y)
        if not (np.isclose(terminal,r.terminal_force_N_total) and np.isclose(terminal/r.loaded_needle_count,observed.at[r.replicate_id,'value'])):
            raise ValueError(f'Terminal force does not reproduce the archived observation in {r.replicate_id}')
        result.append((np.r_[x[x<.8-1e-9],.8], np.r_[y[x<.8-1e-9],terminal], int(r.loaded_needle_count)))
    return result


def terminal_traces(fig, data):
    a = ax(fig,(.085,.274,.50,.419),'both')
    force_values=[]
    for i,((x,y,count),color,style) in enumerate(zip(leader_traces(data),[TEAL,BLUE],['-','--']),1):
        a.plot(x,y,color=color,ls=style,lw=1.8,label=f'Trace {i}')
        a.scatter(.8,y[-1],s=40,color=color,marker='D',edgecolor=PAPER,lw=.7,zorder=4)
        force_values.append(y[-1]/count)
        top=.650-(i-1)*.136
        fig.text(.650,top,f'Trace {i}  ·  {count} loaded needles',fontsize=13.4,weight='bold',color=color)
        fig.text(.650,top-.046,f'{y[-1]:.2f} N ÷ {count} = {force_values[-1]:.3f} N/needle',fontsize=11.8)
    a.axvline(.80,color=GREY,ls=':',lw=1,zorder=1)
    a.set(xlim=(-.015,.85),ylim=(0,79),xticks=[0,.2,.4,.6,.8],xlabel='Displacement from the sustained 1 N trigger (mm)',ylabel='Whole-patch compressive force (N)')
    a.legend(frameon=False,loc='upper left')
    fig.text(.650,.374,'R19 mechanical endpoint',fontsize=13.2,weight='bold')
    fig.text(.647,.291,f'{np.mean(force_values):.3f}',fontsize=27,weight='bold',color=ORANGE)
    fig.text(.769,.301,'N/loaded needle',fontsize=11.9,color=ORANGE)


def assay_plates():
    return [
        Plate('paired-assay',4,'How one patch formulation becomes a measured trade-off',
              'Two measurements describe the same formulation and round. The sketch shows the compression fixture in profile.',
              'After an intact pass, a complete pair joins the observed record. It reaches the frontier if no other measured pair dominates it.',
              'Source • registered MSC viability, intact-formation and terminal-force endpoint definitions · schematic geometry',assay_schematic,'Schematic'),
        Plate('terminal-traces',4,'How the two R19 compression traces give 0.500 N/needle',
              'Both curves come from the ectoin / ethylene glycol / FBS follow-up. Each trace starts at its own sustained 1 N trigger.',
              'At +0.8 mm the two patches produced 68.44 and 31.59 N. Their normalized mean supplies the force coordinate of the R19 frontier point.',
              'Source • Group19_1.csv and Group19_6.csv, verified against both raw-file hashes and canonical observations · terminal_force_08mm_after_1N_v1',terminal_traces)]
