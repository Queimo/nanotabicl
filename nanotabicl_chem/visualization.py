"""Lightweight SVG visualisations for chemical-prior debug artifacts."""

def write_causal_svg(path: str, *, title: str = 'NanoTabICL chemical prior') -> None:
    nodes=[('Molecular structure S',40,90),('latent descriptors B',250,90),('interaction factors I',470,90),('latent state H',690,90),('ideal target Y*',890,90),('observed target Y',1090,90),('observed features X',250,210),('conditions C',690,210),('protocol P',890,210)]
    edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(1,6),(7,2),(7,3),(7,5),(8,5)]
    def box(t,x,y): return f'<rect x="{x}" y="{y}" width="170" height="46" rx="8" fill="#eef6ff" stroke="#2b6cb0"/><text x="{x+85}" y="{y+28}" text-anchor="middle" font-size="13">{t}</text>'
    def line(a,b):
        _,x1,y1=nodes[a]; _,x2,y2=nodes[b]; return f'<line x1="{x1+170}" y1="{y1+23}" x2="{x2}" y2="{y2+23}" stroke="#334155" marker-end="url(#arrow)"/>' if x2>x1 else f'<line x1="{x1+85}" y1="{y1}" x2="{x2+85}" y2="{y2+46}" stroke="#334155" marker-end="url(#arrow)"/>'
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="300" viewBox="0 0 1280 300">','<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#334155"/></marker></defs>',f'<text x="640" y="35" text-anchor="middle" font-size="20" font-weight="bold">{title}</text>']
    svg += [line(a,b) for a,b in edges]
    svg += [box(*n) for n in nodes]
    svg.append('</svg>')
    open(path,'w').write('\n'.join(svg))
