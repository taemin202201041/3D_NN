import json,glob,pandas as pd,numpy as np
from scipy.stats import mannwhitneyu, fisher_exact, beta
df=pd.DataFrame([json.load(open(f)) for f in glob.glob('results/seRNN_pos/*.json')]).drop(columns=['acc_curve'])
df['swap']=df['swap'].fillna(False).astype(bool)
df=df[(df.strength==0.03)&(df.init=='grid')&(df.seed<=40)].copy()
df['cond']=df.name.map(lambda s:'swap' if '_swap_' in s else ('cont' if '_cont_' in s else ('fixed' if '_fixed_' in s else 'old')))
df=df[df.cond.isin(['fixed','swap','cont'])]; df.to_csv('results/seRNN_pos_0.03_3methods.csv',index=False)
def ci(k,n): return (beta.ppf(.025,k,n-k+1) if k>0 else 0, beta.ppf(.975,k+1,n-k) if k<n else 1)
for c in ('fixed','swap','cont'):
    g=df[df.cond==c]; k=int((g.test_acc>0.9).sum()); lo,hi=ci(k,len(g))
    print(f'{c:5} n={len(g)} acc={g.test_acc.mean():.3f}±{g.test_acc.std():.3f} median={g.test_acc.median():.2f} success={k}/{len(g)} ({k/len(g):.0%}, CI {lo:.0%}-{hi:.0%}) disp={g.mean_displacement.mean():.2f} sec={g.seconds.median():.0f}')
A=df[df.cond=='fixed']
for c in ('swap','cont'):
    B=df[df.cond==c]; t=[[int((A.test_acc>0.9).sum()),int((A.test_acc<=0.9).sum())],[int((B.test_acc>0.9).sum()),int((B.test_acc<=0.9).sum())]]
    print(f'fixed vs {c}: acc MWU p={mannwhitneyu(A.test_acc,B.test_acc).pvalue:.3f}  success Fisher p={fisher_exact(t)[1]:.3f}')
L=df[df.test_acc>0.9]
print()
print(L.groupby('cond')[['wd_corr_final','wd_corr_grid','edge_len_final','edge_len_grid','compactness','modularity','smallworld','total_abs_w']].mean().round(3).to_string()); print(L.groupby('cond').size().to_dict())
for met in ('wd_corr_final','edge_len_final','modularity','smallworld'):
    a=L[L.cond=='fixed'][met]; 
    print(met,' | '.join(f'fixed vs {c}: p={mannwhitneyu(a,L[L.cond==c][met]).pvalue:.3g}' for c in ('swap','cont')))
