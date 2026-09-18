import re
t = open('paper/samplepaper.tex', encoding='utf-8').read()
labels = set(re.findall(r'\\label\{([^}]*)\}', t))
refs = set(re.findall(r'\\ref\{([^}]*)\}', t))
print('refs NOT defined:', sorted(refs - labels))
cites = set()
for grp in re.findall(r'\\cite\{([^}]*)\}', t):
    cites |= set(grp.split(','))
bib = set(re.findall(r'\\bibitem\{([^}]*)\}', t))
print('cites NOT in bib:', sorted(cites - bib))
print('bib unused:', sorted(bib - cites))
for term in ['adaptive', 'rewind', 'directional learning', 'EfficientNet+LSTM',
             'sub-daily', '44.4', 'fig:adaptive', 'fig:efficientnet', 'fig:split',
             'fig:model_comparison', 'fig:pixel_intensity', 'tab:results', 'tab:ablation']:
    n = t.count(term)
    if n:
        print(f'still mentions "{term}": {n}x')
