"""Supplementary Instron analysis, with explicit output and frozen settings."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from helper.instron import _read_bluehill_csv
from helper.mechanical_events import export_analysis
from helper.group10_config import load_group10_config
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--force-column',required=True);p.add_argument('--displacement-column',required=True)
    p.add_argument('--needles',required=True,type=int);p.add_argument('--proposal-metadata',type=Path)
    a=p.parse_args()
    cfg=load_group10_config()['mechanical_endpoint']
    if a.proposal_metadata:
        cfg=json.loads(a.proposal_metadata.read_text())['group10']['effective_config']['mechanical_endpoint']
    result=export_analysis(_read_bluehill_csv(a.source),a.force_column,a.displacement_column,cfg,a.needles,a.source,a.output_dir)
    print(result['status']+': '+result['reason'])
