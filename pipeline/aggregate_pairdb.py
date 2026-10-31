#!/usr/bin/env python3
"""
aggregate_pairdb.py — Reproducible aggregation of pipeline output (PairDB_*.csv)
into the de-identified per-examination dataset (Supplementary Data Sheet 1).

Input : PairDB_own.csv / PairDB_shared.csv  (one row per reconstruction; eight
        hemispheric atlas regions: Atlas_<Region>_<L|R>_<metric>)
Output: per-examination table with
        - global metrics  = mean of all successfully measured hemispheric regions (up to 8)
        - regional metrics = mean of the available sides (L, R, or both)
        Profile-based metrics (thickness, table-peak) may be missing in a hemispheric
        region; voxel-based metrics (SDR, cortical/diploic HU) are never missing.
Exclusions (final cohort): age < 18; patients flagged for an implanted device;
        repeat examinations (earliest retained).
De-identification: a code map CSV (folder,code) assigns study codes; identifiers
        are never written to the output.

Usage:
  python3 aggregate_pairdb.py PairDB_own.csv codemap.csv out_own.csv [--exclude-id <registration_id> ...]
Verified 2026-09-21 against the submitted dataset (111 examinations, 4,662 values): all values agree
within one unit of the last reported digit; 4,648 are bit-identical, 14 voxel-based HU means differ by 0.1 HU
owing to floating-point rounding at half-way boundaries in the original export.
"""
import csv, sys, os, argparse, collections
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

def rnd(x, nd):
    return float(np.round(np.float64(x), nd))

REGIONS = ['Frontal', 'Parietal', 'Temporal', 'Occipital']
METRICS = ['SDR', 'Cortex_HU', 'CortexPeak_HU', 'Diploe_HU', 'Outer_mm', 'Diploe_mm', 'Inner_mm', 'Total_mm']
ND = {'SDR': 6, 'Cortex_HU': 1, 'CortexPeak_HU': 1, 'Diploe_HU': 1, 'Outer_mm': 2, 'Diploe_mm': 2, 'Inner_mm': 2, 'Total_mm': 2}

def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def setting(row):
    return 'brain' if '_non_bone_setting' in row['DICOM_Path'] else 'bone'

def folder_of(row):
    return os.path.basename(os.path.dirname(row['DICOM_Path'].rstrip('/')))

def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(np.array(vals, dtype=np.float64))) if vals else None

def aggregate(row):
    out = {}
    for m in METRICS:
        out[m] = mean(fnum(row.get(f'Atlas_{r}_{s}_{m}')) for r in REGIONS for s in 'LR')
        for r in REGIONS:
            out[f'{r}_{m}'] = mean(fnum(row.get(f'Atlas_{r}_{s}_{m}')) for s in 'LR')
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairdb'); ap.add_argument('codemap'); ap.add_argument('out')
    ap.add_argument('--exclude-id', nargs='*', default=[])
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.pairdb, encoding='utf-8-sig')))
    codemap = {r['folder']: r['code'] for r in csv.DictReader(open(a.codemap, encoding='utf-8-sig'))}
    pairs = collections.defaultdict(dict)
    for r in rows:
        pairs[folder_of(r)][setting(r)] = r
    # exclusions
    excl = set()
    by_id = collections.defaultdict(list)
    for fo, d in pairs.items():
        b = d.get('bone') or d.get('brain')
        age = fnum(b['Age'])
        if age is not None and age < 18: excl.add(fo)
        if str(b['Patient_ID']).strip() in set(a.exclude_id): excl.add(fo)
        by_id[str(b['Patient_ID']).strip()].append((b['Study_Date'], fo))
    for pid, lst in by_id.items():
        if len(lst) > 1:
            for _, fo in sorted(lst)[1:]: excl.add(fo)
    hdr = ['study_id', 'age', 'sex', 'scanner', 'kernel_pair', 'otsu_bone', 'otsu_brain']
    for m in METRICS: hdr += [f'{m}_bone', f'{m}_brain']
    for r in REGIONS:
        for m in ['CortexPeak_HU', 'SDR', 'Total_mm']: hdr += [f'{r}_{m}_bone', f'{r}_{m}_brain']
    outrows = []
    for fo, d in pairs.items():
        if fo in excl or 'bone' not in d or 'brain' not in d or fo not in codemap: continue
        b, r = d['bone'], d['brain']; ab, ar = aggregate(b), aggregate(r)
        rec = {'study_id': codemap[fo], 'age': int(fnum(b['Age'])), 'sex': b['Sex'], 'scanner': b['Scanner_Model'],
               'kernel_pair': '', 'otsu_bone': fnum(b['Otsu_HU']), 'otsu_brain': fnum(r['Otsu_HU'])}
        for m in METRICS:
            rec[f'{m}_bone'] = None if ab[m] is None else rnd(ab[m], ND[m]); rec[f'{m}_brain'] = None if ar[m] is None else rnd(ar[m], ND[m])
        for reg in REGIONS:
            for m in ['CortexPeak_HU', 'SDR', 'Total_mm']:
                rec[f'{reg}_{m}_bone'] = None if ab[f'{reg}_{m}'] is None else rnd(ab[f'{reg}_{m}'], 3 if m == 'SDR' else ND[m])
                rec[f'{reg}_{m}_brain'] = None if ar[f'{reg}_{m}'] is None else rnd(ar[f'{reg}_{m}'], 3 if m == 'SDR' else ND[m])
        outrows.append(rec)
    outrows.sort(key=lambda x: x['study_id'])
    with open(a.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
        for rec in outrows: w.writerow({k: ('' if v is None else v) for k, v in rec.items()})
    print(f'pairs {len(pairs)}, excluded {len(excl)}, written {len(outrows)} -> {a.out}')

if __name__ == '__main__':
    main()
