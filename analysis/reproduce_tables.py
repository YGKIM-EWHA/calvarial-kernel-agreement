#!/usr/bin/env python3
"""
reproduce_tables.py — Recompute the agreement statistics reported in the article from the
de-identified per-examination dataset (data/derived_measurements_*.csv).

Reproduces:
  Table 2   automated vs blinded manual total thickness (validation subset, n = 19)
  Table 3   brain- vs bone-kernel agreement, all metrics (n = 111)
  Table 4   stratified agreement and scanner–kernel-specific mapping of table-peak attenuation
  Table S1  shared-threshold sensitivity analysis
  Table S2  SDR category discordance at exploratory cutoffs
  Table S6  stratified agreement for all metrics and between-pair comparisons

Usage:
  python3 analysis/reproduce_tables.py [--data data] [--boot 5000] [--seed 20261031] [--out results]

Bootstrap confidence intervals depend on the random seed; with 5,000 resamples they agree
with the published values to the reported precision, but the last digit may differ.
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats

METRICS = [('SDR', 'SDR', 3), ('Cortex_HU', 'Cortical, voxel-based (HU)', 1),
           ('CortexPeak_HU', 'Table peak (HU)', 1), ('Diploe_HU', 'Diploe (HU)', 1),
           ('Outer_mm', 'Outer table (mm)', 2), ('Diploe_mm', 'Diploe (mm)', 2),
           ('Inner_mm', 'Inner table (mm)', 2), ('Total_mm', 'Total thickness (mm)', 2)]
PAIRS = ['Hr59h/Hr40s', 'H70h/H30s', 'Hr60s/Hr40s', 'Hr68h/Hr38s']
CUTOFFS = [0.40, 0.45, 0.50, 0.55]


def ccc(x, y):
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()               # population variances (ddof = 0)
    return 2 * ((x - mx) * (y - my)).mean() / (vx + vy + (mx - my) ** 2)


def deming(x, y, lam=1.0):
    sxx, syy = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)


def agreement(x, y, rng, nboot, with_deming=True):
    """x = bone (reference), y = brain (or automated vs manual). Bias = y - x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x); d = y - x
    b, s = d.mean(), d.std(ddof=1); t = stats.t.ppf(0.975, n - 1)
    out = dict(n=n, ref_mean=x.mean(), test_mean=y.mean(), bias=b,
               bias_lo=b - t * s / np.sqrt(n), bias_hi=b + t * s / np.sqrt(n),
               loa_lo=b - 1.96 * s, loa_hi=b + 1.96 * s, sd_diff=s,
               r=np.corrcoef(x, y)[0, 1], ccc=ccc(x, y))
    idx = rng.integers(0, n, (nboot, n))
    bc = np.array([ccc(x[i], y[i]) for i in idx])
    out['ccc_lo'], out['ccc_hi'] = np.percentile(bc, [2.5, 97.5])
    if with_deming:
        out['deming'] = deming(x, y)
        bd = np.array([deming(x[i], y[i]) for i in idx])
        out['deming_lo'], out['deming_hi'] = np.percentile(bd, [2.5, 97.5])
    return out


def pair_table(df, rng, nboot, with_deming=True):
    rows = []
    for m, label, _ in METRICS:
        d = df[[f'{m}_bone', f'{m}_brain']].dropna()
        r = agreement(d[f'{m}_bone'], d[f'{m}_brain'], rng, nboot, with_deming)
        r['metric'] = label; rows.append(r)
    return pd.DataFrame(rows).set_index('metric')


def welch_anova(groups):
    k = len(groups); n = np.array([len(g) for g in groups])
    m = np.array([g.mean() for g in groups]); v = np.array([g.var(ddof=1) for g in groups])
    w = n / v; mw = (w * m).sum() / w.sum()
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    tmp = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    f = a / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp); df2 = (k ** 2 - 1) / (3 * tmp)
    return f, df2, stats.f.sf(f, k - 1, df2)


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def loocv_mapping(brain, bone):
    x, y = np.asarray(brain, float), np.asarray(bone, float); n = len(x); pred = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        a, b = np.polyfit(x[keep], y[keep], 1); pred[i] = a * x[i] + b
    a, b = np.polyfit(x, y, 1)
    sse = ((y - pred) ** 2).sum(); sst = ((y - y.mean()) ** 2).sum()
    return a, b, 1 - sse / sst, np.sqrt(sse / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(os.path.dirname(__file__), '..', 'data'))
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=20261031)
    ap.add_argument('--out', default='results')
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    own = pd.read_csv(os.path.join(a.data, 'derived_measurements_own_threshold.csv'))
    shared = pd.read_csv(os.path.join(a.data, 'derived_measurements_shared_threshold.csv'))
    val = pd.read_csv(os.path.join(a.data, 'validation_subset.csv'))
    pd.set_option('display.width', 200); pd.set_option('display.max_columns', 20)

    # Table 2
    t2 = agreement(val['manual_total_mm'], val['automated_total_mm'], rng, a.boot, with_deming=False)
    t2['slope_manual_on_auto'] = np.polyfit(val['automated_total_mm'], val['manual_total_mm'], 1)[0]
    t2 = pd.Series(t2, name='Total thickness'); t2.to_csv(os.path.join(a.out, 'table2_validation.csv'))
    print('\nTable 2 (bias = automated - manual)\n', t2.round(3).to_string())
    reg = val.assign(region=val['region'].str.split(',').str[0]) \
             .groupby('region')['difference_auto_minus_manual_mm'].mean()
    print('Regional offset, manual - automated (mm):\n', (-reg).round(2).to_string())

    # Table 3 and S1
    t3 = pair_table(own, rng, a.boot); t3.to_csv(os.path.join(a.out, 'table3_agreement.csv'))
    print('\nTable 3 (own threshold; bias = brain - bone)\n', t3.round(3).to_string())
    s1 = pair_table(shared, rng, a.boot); s1.to_csv(os.path.join(a.out, 'tableS1_shared_threshold.csv'))
    print('\nTable S1 (shared threshold)\n', s1[['bias', 'ccc']].round(3).to_string())

    # Table 4 and S6a
    rows, maps = [], []
    for p in PAIRS + ['Pooled (n = 111)', 'Pooled excluding J-family']:
        df = own if p.startswith('Pooled (') else own[own.kernel_pair != 'J80s/J30s'] \
            if p.startswith('Pooled excl') else own[own.kernel_pair == p]
        t = pair_table(df, rng, a.boot, with_deming=False); t['stratum'] = p; rows.append(t.reset_index())
        if p in PAIRS:
            sl, ic, r2, rmse = loocv_mapping(df['CortexPeak_HU_brain'], df['CortexPeak_HU_bone'])
            maps.append(dict(stratum=p, n=len(df), slope=sl, intercept=ic, loocv_r2=r2, rmse_hu=rmse))
    s6a = pd.concat(rows); s6a.to_csv(os.path.join(a.out, 'tableS6a_stratified.csv'), index=False)
    sl, ic, r2, rmse = loocv_mapping(own['CortexPeak_HU_brain'][own.kernel_pair.isin(PAIRS)],
                                     own['CortexPeak_HU_bone'][own.kernel_pair.isin(PAIRS)])
    maps.append(dict(stratum='Pooled mapping', n=int(own.kernel_pair.isin(PAIRS).sum()),
                     slope=sl, intercept=ic, loocv_r2=r2, rmse_hu=rmse))
    t4m = pd.DataFrame(maps); t4m.to_csv(os.path.join(a.out, 'table4_mapping.csv'), index=False)
    print('\nTable 4 mapping (bone = slope x brain + intercept)\n', t4m.round(3).to_string(index=False))

    # Table S6b
    het = []
    for m, label, _ in METRICS:
        g = [(own.loc[own.kernel_pair == p, f'{m}_brain'] - own.loc[own.kernel_pair == p, f'{m}_bone']).dropna().values
             for p in PAIRS]
        f, df2, pw = welch_anova(g); kw = stats.kruskal(*g)
        het.append(dict(metric=label, welch_F=f, welch_df2=df2, welch_p=pw, kw_H=kw.statistic, kw_p=kw.pvalue,
                        brown_forsythe_p=stats.levene(*g, center='median').pvalue))
    s6b = pd.DataFrame(het); s6b['welch_p_bonf'] = (s6b.welch_p * len(METRICS)).clip(upper=1)
    s6b['kw_p_bonf'] = (s6b.kw_p * len(METRICS)).clip(upper=1)
    s6b.to_csv(os.path.join(a.out, 'tableS6b_between_pair.csv'), index=False)
    print('\nTable S6b (four principal pairs)\n', s6b.round(4).to_string(index=False))

    # Table S2
    disc = []
    for c in CUTOFFS:
        b, r = own['SDR_bone'] >= c, own['SDR_brain'] >= c
        up, down = int((~b & r).sum()), int((b & ~r).sum()); k = up + down; n = len(own)
        lo, hi = wilson(k, n)
        disc.append(dict(cutoff=c, bone_below=int((~b).sum()), upward=up, downward=down,
                         discordant=k, pct=100 * k / n, wilson_lo=100 * lo, wilson_hi=100 * hi))
    s2 = pd.DataFrame(disc); s2.to_csv(os.path.join(a.out, 'tableS2_sdr_discordance.csv'), index=False)
    s2[['wilson_lo', 'wilson_hi']] = s2[['wilson_lo', 'wilson_hi']].clip(lower=0)
    print('\nTable S2 (SDR category discordance)\n',
          s2.round({'pct': 1, 'wilson_lo': 1, 'wilson_hi': 1}).to_string(index=False))
    print(f'\nResults written to {os.path.abspath(a.out)}')


if __name__ == '__main__':
    main()
