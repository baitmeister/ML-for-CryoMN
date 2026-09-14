"""Versioned terminal extraction and backward-compatible supplementary analysis."""
import hashlib
import json
from pathlib import Path
import numpy as np

REQUIRED=('protocol_id','test_mode','contact_force_N','baseline_points','absolute_drop_N',
          'drop_window_mm','persistence_points','force_sign')

def analyze_curve(force, displacement, config, needles_compressed, time=None):
    from .terminal_force import DEFINITION, analyze_terminal
    if config.get('definition_id') == DEFINITION:
        return analyze_terminal(force, displacement, time, config, needles_compressed)
    result={'definition_id':'supported_load_1mm_v1','detector_version':'force_drop_v1',
            'status':'protocol_incomplete','endpoint_N_total':None,'endpoint_N_per_needle':None,
            'full_window_maximum_N':None,'event_displacement_mm':None,'pre_event_peak_N':None,
            'relative_drop':None,'settings':dict(config),'loaded_needle_count':needles_compressed,
            'interpretation':'nominal N per loaded needle; no event does not imply no failure'}
    missing=[k for k in REQUIRED if config.get(k) is None]
    if missing:
        return {**result,'reason':'Missing settings: '+', '.join(missing)}
    if (config.get('force_unit')!='N' or config.get('displacement_unit')!='mm'
        or config.get('smoothing','none')!='none' or config['test_mode'] not in ('single_needle','array')
        or config['force_sign'] not in (-1,1)):
        return {**result,'reason':'Unsupported units, smoothing, mode or sign'}
    for k in ('baseline_points','persistence_points'):
        if type(config[k]) is not int or config[k]<1: return {**result,'reason':'Invalid '+k}
    for k in ('contact_force_N','absolute_drop_N','drop_window_mm'):
        if not np.isfinite(config[k]) or config[k]<=0: return {**result,'reason':'Invalid '+k}
    if config.get('displacement_limit_mm')!=1.0 or config.get('relative_drop')!=.1:
        return {**result,'reason':'Definition requires 1 mm and 10% drop'}
    if type(needles_compressed) is not int or needles_compressed<=0:
        return {**result,'status':'invalid_curve','reason':'Positive loaded needle count required'}
    if config['test_mode']=='single_needle' and needles_compressed!=1:
        return {**result,'status':'invalid_curve','reason':'Single needle mode requires count 1'}
    f=np.asarray(force,dtype=float)*config['force_sign']; d=np.asarray(displacement,dtype=float)
    if f.ndim!=1 or d.shape!=f.shape or len(f)<3 or not np.isfinite(f).all() or not np.isfinite(d).all():
        return {**result,'status':'invalid_curve','reason':'Invalid arrays or gaps; samples are not stitched'}
    b=config['baseline_points']
    if len(f)<=b: return {**result,'status':'invalid_curve','reason':'Insufficient baseline'}
    f=f-f[:b].mean()
    contacts=np.flatnonzero(f[b:]>=config['contact_force_N'])+b
    if not len(contacts): return {**result,'status':'incomplete_test','reason':'Contact not detected'}
    contact=int(contacts[0]); f=f[contact:]; d=d[contact:]-d[contact]
    result.update(contact_displacement_mm=float(displacement[contact]),contact_index=contact)
    # Stop at first sample reaching 1 mm. Interpolate the endpoint; never use a peak beyond it.
    ends=np.flatnonzero(d>=1.)
    complete=bool(len(ends)); end=int(ends[0]) if complete else len(d)-1
    f=f[:end+1]; d=d[:end+1]
    if np.any(np.diff(d)<0):
        return {**result,'status':'ambiguous_curve','reason':'Unloading before completion'}
    if complete and d[-1]>1:
        f[-1]=np.interp(1.,d[-2:],f[-2:]); d[-1]=1.
    if np.any(f< -config['absolute_drop_N']):
        return {**result,'status':'ambiguous_curve','reason':'Negative corrected load after contact'}
    result.update(full_window_maximum_N=float(np.max(f)),analyzed_displacement_mm=float(d[-1]))
    count=config['persistence_points']
    for i in range(1,len(f)-1):
        if f[i]<=0 or f[i]<f[i-1] or f[i]<=f[i+1]: continue
        required=max(.1*f[i],config['absolute_drop_N'])
        for j in range(i+1,len(f)-count+1):
            if d[j+count-1]-d[i]>config['drop_window_mm']: break
            if np.all(f[i]-f[j:j+count]>=required-1e-12):
                endpoint=float(np.max(f[:i+1]))
                return {**result,'status':'event_detected','reason':'First qualifying force drop',
                        'endpoint_N_total':endpoint,'endpoint_N_per_needle':endpoint/needles_compressed,
                        'event_displacement_mm':float(d[i]),'pre_event_peak_N':float(f[i]),
                        'relative_drop':float((f[i]-min(f[j:j+count]))/f[i])}
    if not complete: return {**result,'status':'incomplete_test','reason':'No event and less than 1 mm'}
    endpoint=float(np.max(f))
    return {**result,'status':'no_event_detected','reason':'Completed 1 mm; no qualifying drop detected',
            'endpoint_N_total':endpoint,'endpoint_N_per_needle':endpoint/needles_compressed}

def export_analysis(frame, force_column, displacement_column, config, needles_compressed, source, output_dir, time_column=None):
    """Explicit output only. Original files and production endpoint are untouched."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=True)
    from .terminal_force import DEFINITION, analyze_frame, numeric_curve
    if config.get('definition_id') == DEFINITION:
        result=analyze_frame(frame,config,needles_compressed,force_column,displacement_column,time_column)
        force_column=result['force_column']; displacement_column=result['displacement_column']
        arrays,_=numeric_curve(frame,[force_column,displacement_column,result['time_column']],config)
        force,displacement=arrays[:2]
    else:
        arrays,_=numeric_curve(frame,[force_column,displacement_column],config)
        force,displacement=arrays
        result=analyze_curve(force,displacement,config,needles_compressed)
    result['source_file_hash']=hashlib.sha256(Path(source).read_bytes()).hexdigest()
    result['source_file']=str(Path(source).resolve())
    result['force_column']=force_column
    result['displacement_column']=displacement_column
    (output/'mechanical_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,ax=plt.subplots(); ax.plot(displacement,force)
    contact=result.get('contact_displacement_mm')
    if contact is not None:
        ax.axvline(contact,color='grey'); ax.axvline(contact+config['displacement_limit_mm'],color='grey',linestyle='--')
        if result['event_displacement_mm'] is not None: ax.axvline(contact+result['event_displacement_mm'],color='red')
        if result.get('terminal_displacement_mm') is not None:
            ax.scatter([result['terminal_displacement_mm']],[result['endpoint_N_total']],color='red',label='F at +0.8 mm')
            ax.legend()
    ax.set(xlabel='Recorded displacement (mm)',ylabel='Recorded force (N)',title=result['status'])
    fig.savefig(output/'mechanical_analysis.png',dpi=160,bbox_inches='tight'); plt.close(fig)
    return result
