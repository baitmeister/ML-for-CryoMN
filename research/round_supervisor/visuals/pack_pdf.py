"""Add a vector contents page, internal links and chapter bookmarks to the figure PDF."""

from io import BytesIO
import json
from pathlib import Path
import sys

from pypdf import PdfReader, PdfWriter
from pypdf.annotations import Link
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas


def contents(width, height, items):
    stream = BytesIO()
    pdf = Canvas(stream, pagesize=(width,height))
    ink, grey, teal, line = '#172f3c', '#526977', '#08766b', '#d4dfdd'
    colors = [teal, '#3c62a2', '#ad4c25', '#72588c', '#38666e']
    pdf.setFillColor(HexColor('#fdfcf9'))
    pdf.rect(0,0,width,height,stroke=0,fill=1)

    def txt(x, y, value, size, color=ink, bold=False):
        pdf.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
        pdf.setFillColor(HexColor(color))
        pdf.drawString(x,y,value)

    txt(51,493,'CRYOMICRONEEDLE FORMULATION RESEARCH',10,teal,True)
    txt(51,449,'A visual guide to the CryoMN round supervisor',32,ink,True)
    txt(52,422,'The goal is to maximize post-thaw mesenchymal stromal cell viability and terminal compression force per loaded needle.',12.3,grey)
    txt(52,405,'Follow seven archived rounds from the frozen predictions to the next research question.',11.5,grey)
    pdf.setStrokeColor(HexColor(line))
    pdf.line(51,394,width-51,394)
    links = []

    def entries(x, y, chapter):
        subset = [i for i in items if i['chapter']==chapter]
        txt(x,y,subset[0]['section'],14,colors[chapter],True)
        for i in subset:
            y -= 16.2
            txt(x+1,y,f"{i['number']:02d}",9.6,colors[chapter],True)
            txt(x+27,y,i['label'],11.0,ink)
            links.append((i['number'],(x,y-4,x+365,y+10.8)))

    entries(52,367,0)
    entries(52,263,1)
    entries(500,367,2)
    entries(500,244,3)
    entries(500,157,4)
    pdf.line(52,89,width-52,89)
    txt(52,65,'Reading route',10,teal,True)
    txt(133,65,'01 for the loop  /  06 for shortlist performance  /  16 for the evidence on steering  /  19 for the pending decision',11,ink)
    drawings=sum(i['kind']=='Schematic' for i in items)
    txt(52,34,f"{drawings} schematics and {len(items)-drawings} data plots. Each title links to a figure. Chapter bookmarks are available in supporting PDF viewers.",9,grey)
    pdf.showPage()
    pdf.save()
    stream.seek(0)
    return stream, links


def pack(input_pdf, output_pdf, items):
    figures = PdfReader(input_pdf)
    if len(figures.pages) != len(items):
        raise ValueError('Figure count differs from the contents page')
    box = figures.pages[0].mediabox
    cover, links = contents(float(box.width),float(box.height),items)
    writer = PdfWriter()
    writer.add_page(PdfReader(cover).pages[0])
    writer.append(figures,import_outline=False)
    for page, rect in links:
        writer.add_annotation(0, Link(rect=rect,target_page_index=page))
    for i in items:
        if i['number']==1 or items[i['number']-2]['chapter']!=i['chapter']:
            parent=writer.add_outline_item(i['section'],i['number'])
        writer.add_outline_item(f"{i['number']:02d}  {i['label']}",i['number'],parent=parent)
    writer.add_metadata({'/Title':'A visual guide to the CryoMN round supervisor',
                         '/Subject':'System schematics, observed R13–R19 measurements, frozen forecast evaluation and the R20 draft'})
    with Path(output_pdf).open('wb') as handle:
        writer.write(handle)


if __name__=='__main__':
    pack(*sys.argv[1:3],json.load(sys.stdin))
