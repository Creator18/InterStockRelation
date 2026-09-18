"""Build news_tests.ipynb - detailed news / first-order / scope diagnostics.

Three questions, in order of build-cost:
  TEST A/B  Does news predict FIRST-ORDER VOLATILITY (not correlation direction)?
            News features: tone level, |tone|, tone change, article volume.
            Target: next-H sector realized volatility. Control for current vol + VIX.
  TEST C    Does EVENT SCOPE drive coupling direction? (the mechanism behind
            'detailed news')  Data-derived scope (no LLM): idiosyncratic shock
            (stock moves, sector doesn't) -> decouple?  sector-wide shock -> couple?
            If this mechanism holds, a news scope-classifier is worth building.
  TEST D    (optional, gated) LLM-scored NEWS scope on real headlines around the
            largest coupling-change events. Requires ANTHROPIC_API_KEY + network.

Everything except Test D runs offline from data already on disk (corr_video,
raw CSVs, cached GDELT tone). Test B also fetches GDELT article volume (cached).
"""
import json

def code(cid, src):
    return {"cell_type":"code","id":cid,"metadata":{},"execution_count":None,
            "outputs":[],"source":src.splitlines(keepends=True)}
def md(cid, src):
    return {"cell_type":"markdown","id":cid,"metadata":{},"source":src.splitlines(keepends=True)}

cells=[]

cells.append(md("md-01", """# News Tests: First-Order Volatility, Event Scope, and Detailed News

Our earlier GDELT test used the crudest signal (daily average tone) against the
hardest target (second-order coupling *direction*) and found nothing. This
notebook tests the two angles that were never fairly tried:

1. **First-order volatility** (Test A/B): does news predict *how much* a sector
   moves, rather than *which way* its correlations shift? Volatility is far more
   predictable than direction, and news is a known volatility driver.
2. **Event scope** (Test C): does the *scope* of a shock (idiosyncratic vs.
   sector-wide) determine coupling *direction*? Idiosyncratic events should
   decouple a stock from its peers; sector-wide events should couple them. We
   test this mechanism from price data (no news, no LLM) --- if it holds, a
   news scope-classifier is worth building.
3. **Detailed news scope** (Test D, optional): LLM-scored scope on real
   headlines around the largest coupling-change events.

Run top to bottom. Tests A--C are offline; Test D is gated behind a flag."""))

cells.append(code("cell-01", """# Cell 1 - Config, imports, sector map
import os, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from numpy.linalg import lstsq

_cwd=os.getcwd()
ROOT=os.path.abspath(os.path.join(_cwd,'..')) if os.path.basename(_cwd)=='notebooks' else _cwd
DATA=os.path.join(ROOT,'data','processed'); RAW=os.path.join(ROOT,'data','raw')
GD=os.path.join(ROOT,'data','external','gdelt')
HORIZON=5; LOOKBACK=20

corr=np.load(os.path.join(DATA,'corr_video.npy'))
pm=pickle.load(open(os.path.join(DATA,'pair_meta.pkl'),'rb'))
vm=pickle.load(open(os.path.join(DATA,'video_meta.pkl'),'rb'))
ts=pm['timestamps']; snap_ts=np.asarray(vm['snap_timesteps'])
ordered=vm['ordered_tickers']; pos={tk:p for p,tk in enumerate(ordered)}
snap_dates=pd.to_datetime(pd.Series(vm['snap_dates'])).dt.normalize()
N_SNAPS,N=corr.shape[0],corr.shape[1]

SECTORS={
 'banking':['AXISBANK','BAJAJFINSV','BAJFINANCE','BANKBARODA','CANBK','CHOLAFIN','HDFCAMC','HDFCBANK','HDFCLIFE','ICICIBANK','KOTAKBANK','MUTHOOTFIN','PFC','PNB','RECLTD','SBILIFE','SBIN','SHRIRAMFIN','UNIONBANK','IRFC'],
 'it':['HCLTECH','INFY','LTM','TCS','TECHM','WIPRO'],
 'pharma':['APOLLOHOSP','CIPLA','DIVISLAB','DRREDDY','MAXHEALTH','SUNPHARMA','TORNTPHARM','ZYDUSLIFE'],
 'auto':['BAJAJ-AUTO','BOSCHLTD','EICHERMOT','MARUTI','MM','MOTHERSON','TVSMOTOR','TMPV'],
 'energy':['ADANIENSOL','ADANIGREEN','ADANIPOWER','BPCL','GAIL','IOC','NTPC','ONGC','POWERGRID','RELIANCE','TATAPOWER','COALINDIA','SOLARINDS'],
}
SECTOR_POS={s:[pos[t] for t in tks if t in pos] for s,tks in SECTORS.items()}
print('snaps',N_SNAPS,'| sectors',{s:len(p) for s,p in SECTOR_POS.items()})
"""))

cells.append(code("cell-02", """# Cell 2 - Fetch GDELT article VOLUME (attention proxy), cached with backoff
# Tone is already cached from the earlier run; here we add volume intensity.
import urllib.request, urllib.parse, ssl
ssl._create_default_https_context=ssl._create_unverified_context
os.makedirs(GD,exist_ok=True)
QUERIES={
 'banking':'(bank OR HDFC OR ICICI OR SBI OR RBI) sourcecountry:IN',
 'it':'(Infosys OR TCS OR Wipro OR software OR "information technology") sourcecountry:IN',
 'pharma':'(pharma OR drug OR Cipla OR medicine OR healthcare) sourcecountry:IN',
 'auto':'(automobile OR Maruti OR "Tata Motors" OR vehicle) sourcecountry:IN',
 'energy':'(oil OR ONGC OR energy OR petroleum) sourcecountry:IN',
}
def fetch_vol(name,query,tries=6):
    fn=os.path.join(GD,f'{name}_timelinevol.csv')
    if os.path.exists(fn) and os.path.getsize(fn)>100: return f'{name}: cached'
    url='https://api.gdeltproject.org/api/v2/doc/doc?'+urllib.parse.urlencode(
        {'query':query,'mode':'timelinevol','startdatetime':'20210701000000','enddatetime':'20260401000000','format':'csv'})
    for k in range(tries):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (research)'})
            with urllib.request.urlopen(req,timeout=60) as r: d=r.read()
            open(fn,'wb').write(d); time.sleep(3); return f'{name}: fetched {len(d)}B'
        except urllib.error.HTTPError as e:
            if e.code==429: time.sleep(8*(k+1))
            else: return f'{name}: ERR {e}'
    return f'{name}: rate-limited (will skip volume for this sector)'
for n,q in QUERIES.items(): print(fetch_vol(n,q))
"""))

cells.append(code("cell-03", """# Cell 3 - Build aligned series: prices/returns, sector realized vol, coupling, news, VIX
# Per-stock snapshot closes in matrix-position (Fiedler) order
close=pd.DataFrame(index=ts)
for tk in ordered:
    d=pd.read_csv(os.path.join(RAW,f'{tk}_5minute.csv'),parse_dates=['date']).set_index('date')
    close[tk]=d['close'].reindex(ts)
sc=close.iloc[snap_ts].to_numpy()                     # (N_SNAPS, 95) ordered
ret=np.vstack([np.full((1,N),np.nan), np.log(sc[1:]/sc[:-1])])   # snap-to-snap returns
mkt_ret=np.nanmean(ret,axis=1)                        # equal-weight market return

def sector_realized_vol(spos, s, H):
    # avg over members of std of returns in (s, s+H]
    w=ret[s+1:s+H+1][:,spos]
    return float(np.nanmean(np.nanstd(w,axis=0)))
def sector_coupling(spos, s):
    sub=corr[s][np.ix_(spos,spos)]; m=~np.eye(len(spos),dtype=bool)
    return float(sub[m].mean())

# News: tone (cached) + volume (Cell 2) aligned to snapshot dates
def load_news(name, mode):
    fn=os.path.join(GD,f'{name}_{mode}.csv')
    if not os.path.exists(fn): return None
    df=pd.read_csv(fn,encoding='utf-8-sig')
    ser=df.iloc[:,1].iloc[0]  # series name in col 1
    d=df[df.iloc[:,1]==df.iloc[:,1].iloc[0]].copy()
    d['Date']=pd.to_datetime(d['Date']).dt.normalize()
    return d.set_index('Date')['Value'].reindex(snap_dates).ffill().bfill().to_numpy().astype(float)

# VIX
vix=pd.read_csv(os.path.join(RAW,'INDIA VIX_5minute.csv'),parse_dates=['date']).set_index('date')['close']
VIX=pd.Series(vix.reindex(ts).to_numpy()[snap_ts]).ffill().bfill().to_numpy()
print('prices/returns ready | mkt_ret std', round(np.nanstd(mkt_ret),4))
print('news available:', {s: (load_news(s,'timelinetone') is not None, load_news(s,'timelinevol') is not None) for s in SECTORS})
"""))

cells.append(code("cell-04", """# Cell 4 - TEST A/B: does news predict next-H sector REALIZED VOLATILITY?
# Control for current volatility (vol clustering) and VIX. Pool across sectors.
s_lo, s_hi = LOOKBACK, N_SNAPS - HORIZON
def resid(y, x):   # residual of y after linear fit on x
    m=~(np.isnan(y)|np.isnan(x)); b=np.polyfit(x[m],y[m],1)
    r=np.full_like(y,np.nan); r[m]=y[m]-np.polyval(b,x[m]); return r

FEATS=['tone','abstone','tonechg','volume']
rows=[]
pooled={k:[] for k in FEATS}; pooled_resid=[]; pooled_vix=[]
for name,spos in SECTOR_POS.items():
    tone=load_news(name,'timelinetone'); vol=load_news(name,'timelinevol')
    fut_vol=np.array([sector_realized_vol(spos,s,HORIZON) for s in range(N_SNAPS)])
    cur_vol=np.array([sector_realized_vol(spos,max(0,s-HORIZON),HORIZON) for s in range(N_SNAPS)])
    sidx=np.arange(s_lo,s_hi); L=len(sidx)
    y=resid(fut_vol[sidx], cur_vol[sidx])   # next-H vol residual (remove clustering)
    # build each feature, padding with NaN when the source series is absent
    fv={
        'tone'   : tone[sidx]                if tone is not None else np.full(L,np.nan),
        'abstone': np.abs(tone[sidx])        if tone is not None else np.full(L,np.nan),
        'tonechg': np.r_[0,np.diff(tone)][sidx] if tone is not None else np.full(L,np.nan),
        'volume' : vol[sidx]                 if vol  is not None else np.full(L,np.nan),
    }
    r={}
    for k in FEATS:
        v=fv[k]; m=~(np.isnan(v)|np.isnan(y))
        r[k]=pearsonr(v[m],y[m])[0] if m.sum()>10 else np.nan
        pooled[k].append(v)             # always append (NaN-padded) -> equal lengths
    pooled_resid.append(y); pooled_vix.append(VIX[sidx])
    rows.append(dict(sector=name, **r))
df=pd.DataFrame(rows)
print('Partial corr of news feature with next-H sector realized-vol residual (per sector):')
print(df.to_string(index=False, float_format='%.3f'))

R=np.concatenate(pooled_resid); V=np.concatenate(pooled_vix)
print('\\nPOOLED partial corr with vol residual:')
for k in FEATS:
    x=np.concatenate(pooled[k]); m=~(np.isnan(x)|np.isnan(R))
    val=f'{pearsonr(x[m],R[m])[0]:+.3f}' if m.sum()>10 else '  n/a (no data)'
    print(f'  {k:<9}: {val}')
# incremental over VIX (does news add beyond VIX for volatility?)
def r2(cols):
    A=np.column_stack(cols+[np.ones(len(R))]); m=~np.isnan(A).any(1)
    if m.sum()<20: return float('nan')
    c,_,_,_=lstsq(A[m],R[m],rcond=None); p=A[m]@c
    return 1-np.sum((R[m]-p)**2)/np.sum((R[m]-R[m].mean())**2)
Vol=np.concatenate(pooled['volume'])
if np.isfinite(Vol).sum()>20:
    print(f'\\n  R^2 VIX only={r2([V]):.4f} | +news volume={r2([V,Vol]):.4f}')
else:
    print('\\n  (news volume not available -- run Cell 2 to fetch it for the incremental test)')
print('\\nVERDICT: a clearly positive pooled corr (esp. volume/|tone|) => news predicts')
print('sector volatility -> worth building a news attention signal for ALERTS.')
"""))

cells.append(code("cell-05", """# Cell 5 - TEST C: does EVENT SCOPE drive coupling direction? (data-derived, no news)
# Idiosyncratic shock (stock moves, sector doesn't) -> stock DECOUPLES from sector?
# Sector-wide shock (whole sector moves) -> sector COUPLES tighter?
csum=np.cumsum(corr,0)

idio_x=[]; idio_resid=[]        # per-(stock,snap): idio shock magnitude vs its coupling change
sec_x=[]; sec_resid=[]          # per-(sector,snap): sector shock vs sector coupling change
for name,spos in SECTOR_POS.items():
    spos=np.array(spos); k=len(spos)
    # sector return per snapshot & member idiosyncratic returns
    sec_ret=np.nanmean(ret[:,spos],axis=1)             # (N_SNAPS,)
    idio=ret[:,spos]-sec_ret[:,None]                    # (N_SNAPS,k) stock-minus-sector
    # per-stock coupling to sector = mean corr with other members
    offm=~np.eye(k,dtype=bool)
    coup=np.array([corr[s][np.ix_(spos,spos)][offm].reshape(k,k-1).mean(1) if False else
                   np.array([corr[s][np.ix_(spos,spos)][i][np.arange(k)!=i].mean() for i in range(k)])
                   for s in range(N_SNAPS)])          # (N_SNAPS,k)
    sec_coup=coup.mean(1)                               # sector-avg coupling
    sidx=np.arange(LOOKBACK, N_SNAPS-HORIZON)
    for s in sidx:
        # idiosyncratic: |idio return into s| vs stock's coupling change residual
        chg=coup[s+HORIZON]-coup[s]
        b=np.polyfit(coup[s], chg, 1); rs=chg-np.polyval(b,coup[s])   # remove mean-rev per snapshot cross-section
        idio_x.append(np.abs(idio[s])); idio_resid.append(rs)
    # sector-level
    schg=sec_coup[sidx+HORIZON]-sec_coup[sidx]
    b=np.polyfit(sec_coup[sidx],schg,1); srs=schg-np.polyval(b,sec_coup[sidx])
    sec_x.append(np.abs(sec_ret[sidx])); sec_resid.append(srs)

IX=np.concatenate(idio_x); IR=np.concatenate(idio_resid)
SX=np.concatenate(sec_x); SR=np.concatenate(sec_resid)
mi=~(np.isnan(IX)|np.isnan(IR)); ms=~(np.isnan(SX)|np.isnan(SR))
print('SCOPE -> COUPLING mechanism (data-derived):')
print(f'  idiosyncratic shock magnitude vs stock coupling-change residual : {pearsonr(IX[mi],IR[mi])[0]:+.3f}')
print(f'     (expect NEGATIVE: stock-specific shock -> stock decouples from sector)')
print(f'  sector-wide shock magnitude   vs sector coupling-change residual: {pearsonr(SX[ms],SR[ms])[0]:+.3f}')
print(f'     (expect POSITIVE: sector-wide shock -> sector couples tighter)')
print('\\nVERDICT: signs as expected + non-trivial magnitude => scope drives coupling,')
print('so a NEWS scope-classifier (idiosyncratic vs sector vs market) is worth building.')
"""))

cells.append(code("cell-06", """# Cell 6 - TEST D (OPTIONAL, gated): LLM-scored NEWS scope on real headlines
# Requires ANTHROPIC_API_KEY and network. Set RUN_LLM_SCOPE=True to enable.
RUN_LLM_SCOPE=False

if RUN_LLM_SCOPE:
    import urllib.request, urllib.parse, ssl, json as _json
    ssl._create_default_https_context=ssl._create_unverified_context
    try:
        import anthropic
    except ImportError:
        raise SystemExit('pip install anthropic, and set ANTHROPIC_API_KEY')
    client=anthropic.Anthropic()

    # Find the largest sector coupling-change events, fetch headlines around them
    def top_events(name,spos,k=15):
        spos=np.array(spos); offm=~np.eye(len(spos),dtype=bool)
        sc=np.array([corr[s][np.ix_(spos,spos)][offm].mean() for s in range(N_SNAPS)])
        chg=np.array([sc[s+HORIZON]-sc[s] if s+HORIZON<N_SNAPS else np.nan for s in range(N_SNAPS)])
        idx=np.argsort(-np.abs(np.nan_to_num(chg)))[:k]
        return [(int(i), float(chg[i]), snap_dates.iloc[i]) for i in idx]

    def gdelt_headlines(query, date, span_days=2, n=5):
        start=(date-pd.Timedelta(days=span_days)).strftime('%Y%m%d000000')
        end=date.strftime('%Y%m%d235959')
        url='https://api.gdeltproject.org/api/v2/doc/doc?'+urllib.parse.urlencode(
            {'query':query,'mode':'artlist','maxrecords':n,'startdatetime':start,'enddatetime':end,'format':'json'})
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
            arts=_json.load(urllib.request.urlopen(req,timeout=60)).get('articles',[])
            return [a.get('title','') for a in arts]
        except Exception: return []

    def score_scope(headlines):
        if not headlines: return None
        prompt=('Classify the market scope of these Indian-market news headlines as one word: '
                'IDIOSYNCRATIC (one company), SECTOR (one sector), or MARKET (whole market). '
                'Reply with just the word.\\n\\n'+'\\n'.join(f'- {h}' for h in headlines))
        m=client.messages.create(model='claude-sonnet-5',max_tokens=10,
                                 messages=[{'role':'user','content':prompt}])
        return m.content[0].text.strip().upper()

    QUERIES={'banking':'(bank OR RBI) sourcecountry:IN','it':'(Infosys OR TCS OR IT) sourcecountry:IN',
             'pharma':'pharma sourcecountry:IN','auto':'(Maruti OR auto) sourcecountry:IN',
             'energy':'(oil OR energy) sourcecountry:IN'}
    results=[]
    for name,spos in SECTOR_POS.items():
        for s,chg,date in top_events(name,spos,10):
            heads=gdelt_headlines(QUERIES[name],date)
            scope=score_scope(heads)
            if scope: results.append(dict(sector=name,date=str(date.date()),chg=chg,scope=scope,n=len(heads)))
            time.sleep(1)
    rdf=pd.DataFrame(results)
    print(rdf.to_string(index=False))
    # Hypothesis: SECTOR/MARKET scope -> coupling UP (chg>0); IDIOSYNCRATIC -> coupling DOWN
    if len(rdf):
        for sc in ['IDIOSYNCRATIC','SECTOR','MARKET']:
            sub=rdf[rdf['scope']==sc]
            if len(sub): print(f'{sc:<14}: mean coupling change {sub[\"chg\"].mean():+.3f} (n={len(sub)})')
else:
    print('Test D disabled. Set RUN_LLM_SCOPE=True with ANTHROPIC_API_KEY to run the')
    print('LLM news-scope test. Test C already tests the same mechanism from price data.')
"""))

cells.append(code("cell-07", """# Cell 7 - Summary
print('SUMMARY OF NEWS TESTS')
print('='*60)
print('Test A/B  news -> sector realized VOLATILITY (first-order magnitude)')
print('          -> see Cell 4 pooled correlations. Positive = build a news')
print('             attention/volatility signal for cluster ALERTS.')
print('Test C    event SCOPE -> coupling direction (data-derived mechanism)')
print('          -> see Cell 5. Expected signs (idio: -, sector: +) with real')
print('             magnitude => a news scope-classifier is worth building.')
print('Test D    LLM news-scope on real headlines (optional, Cell 6).')
print()
print('Decision rule: build the detailed-news pipeline ONLY for the targets that')
print('pass here (volatility and/or scope), NOT for generic tone -> direction,')
print('which both theory and the earlier GDELT test already ruled out.')
"""))

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3.11.9"}},"nbformat":4,"nbformat_minor":5}
with open("notebooks/news_tests.ipynb","w") as f: json.dump(nb,f,indent=1)
print("Wrote notebooks/news_tests.ipynb with",len(cells),"cells")
