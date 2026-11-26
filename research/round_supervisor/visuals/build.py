"""Rebuild the offline visual guide from the published campaign archive."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from .layout import page, plt
from matplotlib.backends.backend_pdf import PdfPages
from .assay import assay_plates

from .charts import decision_plates, performance_plates
from .diagrams import first_plates, last_plates
from .evidence import HERE, load
from .gallery import manifest, write_html


PDF_NAME = 'CryoMN_visual_guide.pdf'


def pdf_interpreter(requested):
    if requested:
        return requested
    if all(importlib.util.find_spec(pkg) for pkg in ('reportlab','pypdf')):
        return sys.executable
    bundled = Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
    if bundled.is_file():
        return str(bundled)
    raise RuntimeError('Select an interpreter with ReportLab and pypdf using --pdf-python.')


def build(output, pdf_python):
    plates = first_plates()+performance_plates()+decision_plates()+last_plates()+assay_plates()
    items = manifest(plates)
    data = load()
    figures = output/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp:
        intermediate = Path(temp)/'figures.pdf'
        with PdfPages(intermediate) as pdf:
            for item,plate in zip(items,plates):
                fig = page(plate,item['number'],len(items))
                plate.draw(fig,data)
                fig.savefig(output/item['image'],dpi=140)
                pdf.savefig(fig)
                plt.close(fig)
        subprocess.run([pdf_python, str(HERE/'pack_pdf.py'), str(intermediate), str(output/PDF_NAME)],
                       input=json.dumps(items), text=True, check=True)
    write_html(output/'index.html',items,data,PDF_NAME)
    return items


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE)
    parser.add_argument('--pdf-python',help='Python executable containing ReportLab and pypdf')
    args = parser.parse_args()
    items = build(args.output, pdf_interpreter(args.pdf_python))
    print(f"{len(items)} figures, the bookmarked PDF and the browser guide are in {args.output}.")
