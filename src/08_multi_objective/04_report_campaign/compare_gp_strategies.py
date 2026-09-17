#!/usr/bin/env python3
"""Audit, freeze, evaluate or redraw GP comparison evidence. Never promotes a GP."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from helper.gp_comparison import audit,freeze,evaluate,load_inputs
from helper.gp_comparison_plots import render

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['audit','freeze','evaluate','render'])
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',default='ROUND_010');p.add_argument('--frozen',type=Path)
    p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int,default=None,help='Latest ingested Group when omitted; retrospective audit only')
    a=p.parse_args()
    if a.mode=='audit':audit(a.root,a.output,a.start,a.end)
    elif a.mode=='freeze':freeze(a.root,a.batch,a.output)
    elif a.mode=='evaluate':
        if a.frozen is None:p.error('--frozen is required')
        _,obs,_=load_inputs(a.root);evaluate(a.frozen,obs,a.output)
    else:render(a.output)
if __name__=='__main__':main()
