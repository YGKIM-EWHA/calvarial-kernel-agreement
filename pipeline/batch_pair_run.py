#!/usr/bin/env python3
"""
batch_pair_run.py
brain_bone_setting_real 의 bone / brain(non_bone) 쌍을 '동일 조건'으로 측정

핵심 설계
  ① EAC 는 bone 에서만 검출하고 brain 에는 그대로 적용
     (soft 커널은 추체골 HU 가 낮아져 detect_eac 의 700~2500 창이 실패함.
      동일 acquisition 이므로 슬라이스 위치가 같아 물리적으로 타당하다.)
     슬라이스 수가 다르면 ImagePositionPatient 의 z 로 대응 슬라이스를 찾는다.

  ②  좌표계(중심·반경)도 bone 것을 brain 에 주입
     (좌표계는 (img>300)&(img<2000) 마스크의 무게중심에서 나오는데 soft 커널은
      두개골 경계가 뭉개져 중심·반경이 달라진다. 고정하지 않으면 두 영상에서
      서로 다른 해부학적 지점을 측정하게 된다.)

  ③ Otsu 를 두 방식으로 각각 실행 — 결과가 다르면 그 자체가 발견이다
       own    : bone/brain 각자의 Otsu   (실제로 soft 만 있는 상황을 재현)
       shared : bone 의 Otsu 를 brain 에도 적용 (순수 커널 효과만 분리)

  ④ 커널명·스캐너·Otsu·EAC 를 별도 CSV(_meta.csv)에 기록
     (save_to_excel 헤더에 커널 컬럼이 없어 예전에 커널명을 잃었던 문제 방지)

  ⑤ 기존 SDR_Database.xlsx 를 건드리지 않고 새 파일에 저장

실행 / Usage
  python3 batch_pair_run.py <input_root>                    # own (default): each series its own Otsu
  python3 batch_pair_run.py <input_root> --otsu shared      # bone Otsu applied to brain series
  python3 batch_pair_run.py <input_root> --limit 5          # first 5 examinations only
  (or set SDR_PAIR_BASE instead of <input_root>)
Outputs PairDB_<mode>.csv/.xlsx and PairDB_<mode>_meta.csv in <input_root>.
"""
import sys, os, csv, time, unicodedata
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import SimpleITK as sitk
    import pydicom
    import sdr_io as V
    import spherical_sdr_module as M
    from spherical_sdr_module import calc_spherical_sdr, calc_spherical_fixed_points
except Exception as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("   sdr_io.py, spherical_sdr_module.py must be in the same folder as this script.")
    sys.exit(1)

# Input root: one sub-folder per examination, each containing '<series>_bone_setting'
# and '<series>_non_bone_setting'. Set with the SDR_PAIR_BASE environment variable or as
# the first positional argument (see usage above).
BASE = os.environ.get('SDR_PAIR_BASE', os.path.join(os.getcwd(), 'brain_bone_setting_real'))
SKIP = {'수술환자', 'bone tumor'}

argv = list(sys.argv[1:])
OTSU_MODE = 'own'
if '--otsu' in argv:
    i = argv.index('--otsu'); OTSU_MODE = argv[i + 1]; del argv[i:i + 2]
LIMIT = None
if '--limit' in argv:
    i = argv.index('--limit'); LIMIT = int(argv[i + 1]); del argv[i:i + 2]
pos = [a for a in argv if not a.startswith('--')]
if pos:
    BASE = os.path.expanduser(pos[0])
if OTSU_MODE not in ('own', 'shared'):
    sys.exit("--otsu 는 own 또는 shared 여야 합니다")

OUT_XLSX = os.path.join(BASE, f'PairDB_{OTSU_MODE}.xlsx')
OUT_META = os.path.join(BASE, f'PairDB_{OTSU_MODE}_meta.csv')
V.EXCEL_PATH = OUT_XLSX


def nfc(s):
    return unicodedata.normalize('NFC', s)


def detect_eac(img):
    """뷰어 /load 라우트와 동일한 추체골 최대단면 EAC 감지 (bone 전용)."""
    Z, Y, X = img.shape
    z0, z1 = int(Z * 0.15), int(Z * 0.55)
    scores = []
    for s_i in range(z0, z1):
        sl = img[s_i]
        lp = sl[int(Y * 0.2):int(Y * 0.8), int(X * 0.05):int(X * 0.32)]
        rp = sl[int(Y * 0.2):int(Y * 0.8), int(X * 0.68):int(X * 0.95)]
        ln = int(np.sum((lp > 700) & (lp < 2500)))
        rn = int(np.sum((rp > 700) & (rp < 2500)))
        tot = ln + rn
        sym = 1.0 - abs(ln - rn) / max(ln + rn, 1)
        scores.append(tot * sym)
    arr = np.array(scores)
    if arr.size and arr.max() > 0:
        cand = np.where(arr > arr.max() * 0.6)[0]
        return int(z0 + cand[0])
    return int(Z * 0.30)


def slice_z(folder):
    """시리즈의 슬라이스별 물리 z 좌표 (sitk 정렬 순서와 동일하게 오름차순)."""
    try:
        r = sitk.ImageSeriesReader()
        ids = r.GetGDCMSeriesIDs(folder)
        names = r.GetGDCMSeriesFileNames(folder, ids[0]) if ids else []
        zs = []
        for n in names:
            ds = pydicom.dcmread(n, stop_before_pixels=True, force=True)
            ipp = getattr(ds, 'ImagePositionPatient', None)
            zs.append(float(ipp[2]) if ipp else None)
        return zs if all(z is not None for z in zs) else None
    except Exception:
        return None


def map_eac(bone_dir, brain_dir, eac_bone, n_bone, n_brain):
    """bone 의 EAC 슬라이스를 brain 의 대응 슬라이스로 변환."""
    if n_bone == n_brain:
        zb, zr = slice_z(bone_dir), slice_z(brain_dir)
        if zb and zr and len(zb) == len(zr):
            if abs(zb[eac_bone] - zr[eac_bone]) < 1.0:
                return eac_bone, 'index(동일)'
            j = int(np.argmin([abs(z - zb[eac_bone]) for z in zr]))
            return j, f'z대응({zb[eac_bone]:.1f}→{zr[j]:.1f})'
        return eac_bone, 'index(가정)'
    zb, zr = slice_z(bone_dir), slice_z(brain_dir)
    if zb and zr:
        j = int(np.argmin([abs(z - zb[eac_bone]) for z in zr]))
        return j, f'z대응(장수 {n_bone}≠{n_brain})'
    # 최후: 비례 배분
    return int(round(eac_bone * (n_brain - 1) / max(n_bone - 1, 1))), '비례(z 없음)'


def measure(folder, otsu_override=None):
    """한 시리즈 로드 후 측정에 필요한 값 반환."""
    ok, msg = V.load_dicom(folder)
    if not ok or V.current.get('img') is None:
        raise RuntimeError(f'load 실패: {msg}')
    img = V.current['img']
    otsu_own = float(V.current['otsu'])
    otsu_use = float(otsu_override) if otsu_override is not None else otsu_own
    meta = dict(V.current.get('meta', {}))
    spacing = V.current.get('spacing', (0.5, 0.5, 2.0))
    return img, otsu_own, otsu_use, meta, spacing


def raw_geom(img, eac):
    """그 영상이 '스스로' 계산했을 좌표계 — 진단용(주입값과의 차이 기록)."""
    c = img[eac:, :, :]
    m = (c > 300) & (c < 2000)
    if not np.any(m):
        return None
    z_i, y_i, x_i = np.where(m)
    cy, cx = float(np.mean(y_i)), float(np.mean(x_i))
    r = float(np.percentile(np.sqrt((y_i-cy)**2 + (x_i-cx)**2), 82))
    return cy, cx, float(np.mean(z_i)), r


def kernel_of(folder):
    try:
        for f in sorted(os.listdir(folder)):
            if f.startswith('.'):
                continue
            ds = pydicom.dcmread(os.path.join(folder, f),
                                 stop_before_pixels=True, force=True)
            k = getattr(ds, 'ConvolutionKernel', '')
            if isinstance(k, (list, tuple)) or k.__class__.__name__ == 'MultiValue':
                k = '\\'.join(str(x) for x in k)
            return str(k).split('\\')[0].strip()
    except Exception:
        pass
    return ''


# ── 실행 ───────────────────────────────────────
if not os.path.isdir(BASE):
    sys.exit(f"❌ 폴더 없음: {BASE}")

pats = sorted(d for d in os.listdir(BASE)
              if os.path.isdir(os.path.join(BASE, d))
              and not d.startswith('.') and nfc(d) not in SKIP)
if LIMIT:
    pats = pats[:LIMIT]

# 기존 출력 백업
ts = time.strftime('%Y%m%d_%H%M%S')
for p in (OUT_XLSX, OUT_XLSX.replace('.xlsx', '.csv'), OUT_META):
    if os.path.exists(p):
        os.rename(p, f'{p}.bak_{ts}')
        print(f"🗄️  백업: {os.path.basename(p)}.bak_{ts}")

print(f"\n📂 {BASE}")
print(f"🧮 환자 {len(pats)}명 × 2(bone/brain) = {len(pats)*2} 시리즈")
print(f"⚙️  Otsu 모드: {OTSU_MODE}"
      f"{'  (bone 의 Otsu 를 brain 에도 적용)' if OTSU_MODE=='shared' else '  (각자 Otsu)'}")
print("=" * 88)

meta_rows, ok, fail, failed = [], 0, 0, []
t0 = time.time()

for i, d in enumerate(pats, 1):
    pdir = os.path.join(BASE, d)
    subs = os.listdir(pdir)
    bone_d = next((s for s in subs if s.endswith('_bone_setting')
                   and not s.endswith('_non_bone_setting')), None)
    brain_d = next((s for s in subs if s.endswith('_non_bone_setting')), None)
    if not bone_d or not brain_d:
        fail += 1; failed.append(f'{d}(쌍 없음)')
        print(f"[{i:3d}/{len(pats)}] ❌ {d:26s} bone/brain 쌍 불완전")
        continue

    bone_p = os.path.join(pdir, bone_d)
    brain_p = os.path.join(pdir, brain_d)
    try:
        # ── bone ──
        img_b, otsu_b_own, otsu_b, meta_b, sp_b = measure(bone_p)
        eac_b = detect_eac(img_b)
        reg_b, _ = calc_spherical_sdr(img_b, otsu_b, eac_b)
        geom_sdr = M.LAST_SDR_GEOM                      # (cy,cx,cz)
        fp_b = calc_spherical_fixed_points(img_b, otsu_b, eac_b, sp_b)
        geom_fp = dict(M.LAST_FP_GEOM)                  # 중심·반경·z비율
        V.current['eac_slice'] = eac_b
        V.current['regional_sdr'] = reg_b
        V.save_to_excel(d, bone_p, otsu_b, eac_b, reg_b, fp_b)
        n_bone = img_b.shape[0]

        # ── brain ──  EAC 는 bone 것을 사용
        ov = otsu_b if OTSU_MODE == 'shared' else None
        img_r, otsu_r_own, otsu_r, meta_r, sp_r = measure(brain_p, otsu_override=ov)
        n_brain = img_r.shape[0]
        eac_r, how = map_eac(bone_p, brain_p, eac_b, n_bone, n_brain)
        g_own = raw_geom(img_r, eac_r)               # brain 이 스스로 잡았을 좌표계(진단)
        reg_r, _ = calc_spherical_sdr(img_r, otsu_r, eac_r, geom=geom_sdr)
        fp_r = calc_spherical_fixed_points(img_r, otsu_r, eac_r, sp_r, geom=geom_fp)
        dcy = dcx = dr = ''
        if g_own:
            dcy = round(g_own[0] - geom_fp['cy'], 1)
            dcx = round(g_own[1] - geom_fp['cx'], 1)
            dr = round(g_own[3] - geom_fp['r_xy'], 1)
        V.current['eac_slice'] = eac_r
        V.current['regional_sdr'] = reg_r
        V.save_to_excel(d, brain_p, otsu_r, eac_r, reg_r, fp_r)

        for tag, folder, oo, ou, ea, ns, sp, mt in (
                ('bone', bone_p, otsu_b_own, otsu_b, eac_b, n_bone, sp_b, meta_b),
                ('brain', brain_p, otsu_r_own, otsu_r, eac_r, n_brain, sp_r, meta_r)):
            meta_rows.append({
                'folder': d, 'setting': tag,
                'patient_id': mt.get('patient_id', ''), 'study_date': mt.get('study_date', ''),
                'age': mt.get('age', ''), 'sex': mt.get('sex', ''),
                'scanner': mt.get('model', ''), 'kvp': mt.get('kvp', ''),
                'kernel': kernel_of(folder),
                'otsu_own': round(oo, 1), 'otsu_used': round(ou, 1),
                'eac_slice': ea, 'n_slices': ns,
                'spacing_z': sp[2] if len(sp) > 2 else '',
                'eac_mapping': how if tag == 'brain' else 'detected',
                'geom': '주입(bone)' if tag == 'brain' else '자체계산',
                'geom_dcy': dcy if tag == 'brain' else '',
                'geom_dcx': dcx if tag == 'brain' else '',
                'geom_dr_xy': dr if tag == 'brain' else '',
                'otsu_mode': OTSU_MODE, 'path': folder,
            })

        vb = [v.get('sdr') for v in reg_b.values() if v.get('sdr')]
        vr = [v.get('sdr') for v in reg_r.values() if v.get('sdr')]
        mb = round(sum(vb) / len(vb), 3) if vb else float('nan')
        mr = round(sum(vr) / len(vr), 3) if vr else float('nan')
        ok += 1
        print(f"[{i:3d}/{len(pats)}] ✅ {d:26s} EAC={eac_b:>3d}/{eac_r:<3d} "
              f"Otsu={otsu_b:.0f}/{otsu_r:.0f}  SDR={mb}/{mr}  "
              f"기하편차 cy{dcy:+} r{dr:+}")
    except Exception as e:
        fail += 1; failed.append(f'{d}({e})')
        print(f"[{i:3d}/{len(pats)}] ❌ {d:26s} {e}")

if meta_rows:
    with open(OUT_META, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        w.writeheader(); w.writerows(meta_rows)

print("=" * 88)
print(f"📊 성공 {ok} / 실패 {fail}   (소요 {(time.time()-t0)/60:.1f}분)")
print(f"📄 측정 DB : {OUT_XLSX}")
print(f"📄 메타    : {OUT_META}")
if failed:
    print(f"⚠️  실패: {'; '.join(failed[:15])}")

if meta_rows:
    import collections
    br = [r for r in meta_rows if r['setting'] == 'brain']
    print("\nEAC 대응 방식:", dict(collections.Counter(r['eac_mapping'].split('(')[0] for r in br)))
    bo = [r for r in meta_rows if r['setting'] == 'bone']
    if bo and br:
        db = np.mean([r['otsu_own'] for r in bo]); dr = np.mean([r['otsu_own'] for r in br])
        print(f"Otsu 평균 — bone {db:.0f} / brain {dr:.0f}  (차이 {dr-db:+.0f} HU)")
    dd = [r for r in br if r['geom_dcy'] != '']
    if dd:
        import statistics as st
        print(f"좌표계 편차(brain 자체계산 − bone 주입): "
              f"cy {st.mean([r['geom_dcy'] for r in dd]):+.1f}px, "
              f"r_xy {st.mean([r['geom_dr_xy'] for r in dd]):+.1f}px  "
              f"← 주입하지 않았다면 이만큼 어긋난 지점을 비교했을 것")
    print("커널쌍:", dict(collections.Counter(
        f"{b['kernel']}↔{r['kernel']}" for b, r in zip(bo, br))))
