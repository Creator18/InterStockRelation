import re
t = open('paper/samplepaper.tex', encoding='utf-8').read()
lines = t.split('\n')

print("=== 1. LONG / UNBREAKABLE INLINE MATH ($...$) ===")
# inline math segments
for i, ln in enumerate(lines, 1):
    # strip displayed-math lines
    for m in re.finditer(r'(?<!\\)\$([^$]+)\$', ln):
        body = m.group(1)
        risky = ('\\left' in body or '\\right' in body or '\\frac' in body
                 or '\\sum' in body or '\\mathbb{E}' in body)
        if len(body) > 45 or (risky and len(body) > 30):
            print(f"  L{i}: len={len(body)} risky={risky}  ${body[:90]}$")

print("\n=== 2. LONG DISPLAYED-EQUATION LINES (raw > 90 chars inside eq envs) ===")
in_eq = False
for i, ln in enumerate(lines, 1):
    if re.search(r'\\begin\{(equation|align|align\*|multline|split|gather)\}', ln):
        in_eq = True
    if in_eq and len(ln.rstrip()) > 90 and '\\caption' not in ln:
        print(f"  L{i}: {len(ln.rstrip())} chars | {ln.strip()[:100]}")
    if re.search(r'\\end\{(equation|align|align\*|multline|split|gather)\}', ln):
        in_eq = False

print("\n=== 3. TABLES: column spec + longest cell text ===")
for m in re.finditer(r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}', t, re.S):
    spec, body = m.group(1), m.group(2)
    ncol = sum(c in 'lcr|p' for c in spec.replace('p',''))  # rough
    rows = [r for r in body.split('\\\\') if '&' in r]
    longest = max((len(r.strip()) for r in rows), default=0)
    # count text columns (l/c/r)
    textcols = sum(1 for c in spec if c in 'lcr')
    line_no = t[:m.start()].count('\n') + 1
    print(f"  table near L{line_no}: spec='{spec}' textcols={textcols} longest_row={longest} chars")
    if longest > 95:
        worst = max(rows, key=lambda r: len(r.strip()))
        print(f"      WIDEST ROW: {worst.strip()[:120]}")

print("\n=== 4. \\texttt / \\url / long unbreakable tokens ===")
for i, ln in enumerate(lines, 1):
    for m in re.finditer(r'\\(texttt|url|path)\{([^}]*)\}', ln):
        if len(m.group(2)) > 20:
            print(f"  L{i}: \\{m.group(1)}{{{m.group(2)}}}")
    # very long words (>30 non-space chars)
    for w in re.findall(r'\S{35,}', ln):
        if '\\includegraphics' not in w and 'http' not in w:
            print(f"  L{i}: long token ({len(w)}): {w[:60]}")
