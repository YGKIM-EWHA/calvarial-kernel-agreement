import numpy as np
from skimage.filters import threshold_otsu

def detect_eac_slice(img):
    Z, Y, X = img.shape
    z_min = int(Z * 0.15)
    z_max = int(Z * 0.55)
    scores = []
    for s in range(z_min, z_max):
        sl = img[s]
        lp = sl[int(Y*0.20):int(Y*0.80), int(X*0.05):int(X*0.32)]
        rp = sl[int(Y*0.20):int(Y*0.80), int(X*0.68):int(X*0.95)]
        ln = int(np.sum((lp > 700) & (lp < 2500)))
        rn = int(np.sum((rp > 700) & (rp < 2500)))
        total = ln + rn
        sym = 1.0 - abs(ln - rn) / max(ln + rn, 1)
        scores.append(total * sym)
    scores = np.array(scores)
    if scores.max() > 0:
        thresh = scores.max() * 0.6
        cands = np.where(scores > thresh)[0]
        eac_slice = int(z_min + cands[0])
        method = f"추체골 최대단면 방식 (Slice {eac_slice}, score={scores[cands[0]]:.0f})"
    else:
        eac_slice = int(Z * 0.30)
        method = f"추체골 미감지 기본값 (Slice {eac_slice})"
    return eac_slice, method

def assign_spherical_region(dz, dy, dx):
    r = np.sqrt(dz**2 + dy**2 + dx**2)
    if r < 1e-6:
        return None
    theta = np.degrees(np.arccos(np.clip(dz / r, -1, 1)))
    phi   = np.degrees(np.arctan2(dx, dy))
    if theta < 55:
        return '두정골_좌' if dx >= 0 else '두정골_우'
    elif theta < 100:
        if   -50 < phi <= 50:
            return '전두골_좌' if dx >= 0 else '전두골_우'
        elif  50 < phi <= 130:
            return '측두골_좌'
        elif phi > 130 or phi <= -130:
            return '후두골_좌' if dx >= 0 else '후두골_우'
        else:
            return '측두골_우'
    return None

LAST_SDR_GEOM = None      # 직전 호출에서 사용한 (cy, cx, cz) — 쌍 비교 시 재사용용


def calc_spherical_sdr(img, otsu, eac_slice, geom=None):
    """geom=(cy,cx,cz) 를 주면 그 좌표계를 그대로 사용한다.

    ★ 커널 쌍 비교 시 필수: 좌표계는 (img>300)&(img<2000) 마스크의 무게중심으로
      정해지는데, soft 커널은 두개골 경계가 뭉개져 이 마스크가 달라진다.
      좌표계를 고정하지 않으면 두 영상에서 '서로 다른 부위'를 비교하게 된다."""
    global LAST_SDR_GEOM
    Z, Y, X = img.shape
    img_c = img[eac_slice:, :, :]
    bone_mask = (img_c > 300) & (img_c < 2000)
    if not np.any(bone_mask):
        return {}, "뼈 없음"
    z_i, y_i, x_i = np.where(bone_mask)
    if geom is not None:
        cy, cx, cz = float(geom[0]), float(geom[1]), float(geom[2])
    else:
        cy = float(np.mean(y_i))
        cx = float(np.mean(x_i))
        cz = float(np.mean(z_i))
    LAST_SDR_GEOM = (cy, cx, cz)
    dz = (z_i - cz).astype(float)
    dy = (cy - y_i).astype(float)
    dx = (x_i - cx).astype(float)
    hu = img_c[z_i, y_i, x_i]
    r  = np.sqrt(dz**2 + dy**2 + dx**2)
    ok = r > 1e-6
    theta = np.full(len(z_i), 999.0)
    phi   = np.zeros(len(z_i))
    theta[ok] = np.degrees(np.arccos(np.clip(dz[ok]/r[ok], -1, 1)))
    phi[ok]   = np.degrees(np.arctan2(dx[ok], dy[ok]))
    t = theta; p = phi; base = t < 100
    masks = {
        '두정골_좌': (t < 55) & (dx >= 0),
        '두정골_우': (t < 55) & (dx <  0),
        '전두골_좌': base & (t >= 55) & (p > -50)  & (p <= 50)  & (dx >= 0),
        '전두골_우': base & (t >= 55) & (p > -50)  & (p <= 50)  & (dx <  0),
        '측두골_좌': base & (t >= 55) & (p > 50)   & (p <= 130),
        '측두골_우': base & (t >= 55) & (p > -130) & (p <= -50),
        '후두골_좌': base & (t >= 55) & ((p > 130) | (p <= -130)) & (dx >= 0),
        '후두골_우': base & (t >= 55) & ((p > 130) | (p <= -130)) & (dx <  0),
    }
    colors = {'전두골_좌':'#4FC3F7','전두골_우':'#0288D1','두정골_좌':'#81C784','두정골_우':'#388E3C','측두골_좌':'#FFB74D','측두골_우':'#F57C00','후두골_좌':'#CE93D8','후두골_우':'#7B1FA2'}
    results = {}
    for rname, mask in masks.items():
        hu_r = hu[mask]
        if len(hu_r) < 50:
            results[rname] = {'sdr':None,'cortex_hu':None,'diploe_hu':None,'n':len(hu_r),'color':colors[rname]}
            continue
        cortex = float(np.mean(hu_r[hu_r > otsu])) if np.any(hu_r > otsu) else None
        diploe = float(np.mean(hu_r[hu_r <= otsu])) if np.any(hu_r <= otsu) else None
        sdr    = round(diploe/cortex, 3) if (cortex and diploe) else None
        results[rname] = {'sdr':sdr,'cortex_hu':round(cortex,1) if cortex else None,'diploe_hu':round(diploe,1) if diploe else None,'n':len(hu_r),'color':colors[rname]}
    return results, "OK"

POINT_TARGETS = {
    '전두골_좌': [{'name':'F-L1','desc':'전두골 전상방','theta':78,'phi':15},{'name':'F-L2','desc':'전두골 전방','theta':85,'phi':8},{'name':'F-L3','desc':'전두골 중앙','theta':72,'phi':28},{'name':'F-L4','desc':'전두골 후상방','theta':67,'phi':22},{'name':'F-L5','desc':'전두골 외측상','theta':80,'phi':40}],
    '전두골_우': [{'name':'F-R1','desc':'전두골 전상방','theta':78,'phi':-15},{'name':'F-R2','desc':'전두골 전방','theta':85,'phi':-8},{'name':'F-R3','desc':'전두골 중앙','theta':72,'phi':-28},{'name':'F-R4','desc':'전두골 후상방','theta':67,'phi':-22},{'name':'F-R5','desc':'전두골 외측상','theta':80,'phi':-40}],
    '두정골_좌': [{'name':'P-L1','desc':'두정골 전부','theta':38,'phi':20},{'name':'P-L2','desc':'두정골 중앙','theta':28,'phi':35},{'name':'P-L3','desc':'두정골 후부','theta':40,'phi':55},{'name':'P-L4','desc':'두정골 외측','theta':50,'phi':45},{'name':'P-L5','desc':'두정골 내측상','theta':18,'phi':15}],
    '두정골_우': [{'name':'P-R1','desc':'두정골 전부','theta':38,'phi':-20},{'name':'P-R2','desc':'두정골 중앙','theta':28,'phi':-35},{'name':'P-R3','desc':'두정골 후부','theta':40,'phi':-55},{'name':'P-R4','desc':'두정골 외측','theta':50,'phi':-45},{'name':'P-R5','desc':'두정골 내측상','theta':18,'phi':-15}],
    '측두골_좌': [{'name':'T-L1','desc':'측두골 상전부','theta':82,'phi':68},{'name':'T-L2','desc':'측두골 상중부','theta':87,'phi':88},{'name':'T-L3','desc':'측두골 중앙','theta':90,'phi':102},{'name':'T-L4','desc':'측두골 상후부','theta':85,'phi':118},{'name':'T-L5','desc':'측두골 하전부','theta':95,'phi':72}],
    '측두골_우': [{'name':'T-R1','desc':'측두골 상전부','theta':82,'phi':-68},{'name':'T-R2','desc':'측두골 상중부','theta':87,'phi':-88},{'name':'T-R3','desc':'측두골 중앙','theta':90,'phi':-102},{'name':'T-R4','desc':'측두골 상후부','theta':85,'phi':-118},{'name':'T-R5','desc':'측두골 하전부','theta':95,'phi':-72}],
    '후두골_좌': [{'name':'O-L1','desc':'후두골 상부','theta':63,'phi':152},{'name':'O-L2','desc':'후두골 상중부','theta':70,'phi':162},{'name':'O-L3','desc':'후두골 중앙','theta':78,'phi':155},{'name':'O-L4','desc':'후두골 외측','theta':73,'phi':142},{'name':'O-L5','desc':'람다봉합 후방','theta':58,'phi':158}],
    '후두골_우': [{'name':'O-R1','desc':'후두골 상부','theta':63,'phi':-152},{'name':'O-R2','desc':'후두골 상중부','theta':70,'phi':-162},{'name':'O-R3','desc':'후두골 중앙','theta':78,'phi':-155},{'name':'O-R4','desc':'후두골 외측','theta':73,'phi':-142},{'name':'O-R5','desc':'람다봉합 후방','theta':58,'phi':-158}],
}
SPHERICAL_FIXED_POINTS = POINT_TARGETS

def _ray_cast_bone(img, eac_slice, cy, cx, cz_rel, ux, uy, uz, r_xy, z_ratio, Z, Y, X):
    cz_abs = eac_slice + cz_rel
    for step in range(8, int(r_xy * 1.5), 2):
        ny = int(cy    + uy * step)
        nx = int(cx    + ux * step)
        nz = int(cz_abs + uz * step * z_ratio)
        nz = max(eac_slice, min(Z-1, nz))
        ny = max(0, min(Y-1, ny))
        nx = max(0, min(X-1, nx))
        if 300 <= img[nz, ny, nx] <= 2000:
            r2 = 8
            roi = img[max(0,nz-1):min(Z,nz+2),max(0,ny-r2):min(Y,ny+r2),max(0,nx-r2):min(X,nx+r2)]
            if np.sum((roi > 300) & (roi < 2000)) >= 12:
                return (ny, nx, nz)
    return None

def _calc_sdr_roi(img, otsu, s, y, x, r=15):
    Z, Y, X = img.shape
    s = max(0, min(Z-1, s))
    roi = img[max(0,s-1):min(Z,s+2),max(0,y-r):min(Y,y+r),max(0,x-r):min(X,x+r)]
    bone = roi[(roi > 300) & (roi < 2000)]
    if len(bone) < 20:
        return None, None, None, len(bone)
    cortex = float(np.mean(bone[bone > otsu])) if np.any(bone > otsu) else None
    diploe = float(np.mean(bone[bone <= otsu])) if np.any(bone <= otsu) else None
    sdr    = round(diploe/cortex, 3) if (cortex and diploe) else None
    return sdr, cortex, diploe, len(bone)

LAST_FP_GEOM = None       # 직전 호출의 {cy,cx,cz_rel,r_xy,z_ratio}


def calc_spherical_fixed_points(img, otsu, eac_slice, spacing=(0.5, 0.5, 2.0), geom=None):
    """geom={'cy','cx','cz_rel','r_xy','z_ratio'} 를 주면 그 좌표계를 그대로 사용.

    ★ 커널 쌍 비교 시 필수 — 위 calc_spherical_sdr 의 설명과 동일한 이유."""
    global LAST_FP_GEOM
    Z, Y, X = img.shape
    img_c = img[eac_slice:, :, :]
    bone_m = (img_c > 300) & (img_c < 2000)
    if not np.any(bone_m):
        return {}
    z_i, y_i, x_i = np.where(bone_m)
    if geom is not None:
        cy = float(geom['cy']); cx = float(geom['cx'])
        cz_rel = float(geom['cz_rel']); r_xy = float(geom['r_xy'])
        z_ratio = float(geom['z_ratio'])
        src = ' (주입)'
    else:
        cy  = float(np.mean(y_i))
        cx  = float(np.mean(x_i))
        cz_rel = float(np.mean(z_i))
        xy_d = np.sqrt((y_i - cy)**2 + (x_i - cx)**2)
        r_xy = float(np.percentile(xy_d, 82))
        z_extent  = float(z_i.max() - z_i.min())
        xy_extent = float(max(x_i.max()-x_i.min(), y_i.max()-y_i.min()))
        z_ratio = (z_extent / xy_extent) if xy_extent > 0 else 0.24
        src = ''
    LAST_FP_GEOM = {'cy': cy, 'cx': cx, 'cz_rel': cz_rel,
                    'r_xy': r_xy, 'z_ratio': z_ratio}
    print(f"[FP] cy={cy:.0f} cx={cx:.0f} cz_rel={cz_rel:.0f} r_xy={r_xy:.0f} z_ratio={z_ratio:.3f}{src}")
    results = {}
    for region_name, points in POINT_TARGETS.items():
        results[region_name] = []
        for pt in points:
            theta_r = np.radians(pt['theta'])
            phi_r   = np.radians(pt['phi'])
            ux =  np.sin(theta_r) * np.sin(phi_r)
            uy = -np.sin(theta_r) * np.cos(phi_r)
            uz =  np.cos(theta_r)
            found = _ray_cast_bone(img, eac_slice, cy, cx, cz_rel, ux, uy, uz, r_xy, z_ratio, Z, Y, X)
            if found is None:
                for dt in [10, -10, 15, -15, 20, -20]:
                    t2 = np.radians(pt['theta'] + dt)
                    found = _ray_cast_bone(img, eac_slice, cy, cx, cz_rel,
                                           np.sin(t2)*np.sin(phi_r),
                                           -np.sin(t2)*np.cos(phi_r),
                                           np.cos(t2), r_xy, z_ratio, Z, Y, X)
                    if found: break
            if found:
                fy, fx, fs = found
                sdr, cortex, diploe, n = _calc_sdr_roi(img, otsu, fs, fy, fx, r=15)
                # 동일 위치에서 3D 법선 프로파일로 층별 두께 추출
                lay = measure_normal_layers(img, otsu, fs, fy, fx, spacing)
            else:
                fy, fx, fs = -1, -1, -1
                sdr, cortex, diploe, n = None, None, None, 0
                lay = None
            rec = {'name':pt['name'],'desc':pt['desc'],'theta':pt['theta'],'phi':pt['phi'],'sdr':sdr,'cortex_hu':round(cortex,1) if cortex else None,'diploe_hu':round(diploe,1) if diploe else None,'n':n,'slice':fs,'y_px':fy,'x_px':fx}
            if lay is not None:
                rec.update({
                    'outer_mm': lay['outer_mm'], 'diploe_mm': lay['diploe_mm'],
                    'inner_mm': lay['inner_mm'], 'total_mm': lay['total_mm'],
                    'cortex_peak_hu': lay['cortex_hu'], 'oblique_deg': lay['oblique_deg'],
                })
            else:
                rec.update({'outer_mm':None,'diploe_mm':None,'inner_mm':None,
                            'total_mm':None,'cortex_peak_hu':None,'oblique_deg':None})
            results[region_name].append(rec)
    return results

def calc_sdr_roi_local(img, otsu, s, y, x, r=15):
    return _calc_sdr_roi(img, otsu, s, y, x, r)

def find_nearest_bone(img, s, y, x, r_search=80):
    Z, Y, X = img.shape
    s=max(0,min(Z-1,s)); y=max(0,min(Y-1,y)); x=max(0,min(X-1,x))
    if 400 <= img[s,y,x] <= 2000: return y, x, s
    cy, cx_c = Y//2, X//2
    dx=x-cx_c; dy_c=y-cy
    dist=max(1.0,np.sqrt(dx**2+dy_c**2))
    for step in range(3,r_search,3):
        ny=max(0,min(Y-1,int(y+dy_c/dist*step))); nx=max(0,min(X-1,int(x+dx/dist*step)))
        if 400 <= img[s,ny,nx] <= 2000: return ny, nx, s
    return y, x, s

print("구면 좌표계 SDR 모듈 v2 로드 완료")


# ─────────────────────────────────────────────────────────────
# 3D 법선 프로파일 기반 층별 두께 측정 (외판/해면/내판/전체)
#   - sdr_viewer_v2.calc_normal_sdr 와 동일 알고리즘(자립형)
#   - 각 고정점 위치(s,y,x)에서 외판 표면의 실제 3D 법선을 추정하고
#     슬라이스를 가로질러 trilinear 보간 프로파일을 떠서 층 두께 추출
#   - spacing = SimpleITK GetSpacing() = (sx, sy, sz)  [mm/voxel]
# ─────────────────────────────────────────────────────────────
def measure_normal_layers(img, otsu, s, y, x, spacing=(0.5, 0.5, 2.0)):
    try:
        from scipy.ndimage import map_coordinates
        from scipy.signal import savgol_filter
        has_scipy = True
    except Exception:
        has_scipy = False

    Z, Y, X = img.shape
    s = int(max(1, min(Z - 2, s)))
    y = int(max(2, min(Y - 3, y)))
    x = int(max(2, min(X - 3, x)))
    sp = spacing or (0.5, 0.5, 2.0)
    sx = float(sp[0]) or 0.5
    sy = float(sp[1]) or 0.5
    sz = float(sp[2]) or 2.0

    # 1) 시드 법선: 국소 3D 그래디언트(물리좌표)
    gz = (float(img[min(Z-1, s+1), y, x]) - float(img[max(0, s-1), y, x])) / (2 * sz)
    gy = (float(img[s, min(Y-1, y+2), x]) - float(img[s, max(0, y-2), x])) / (4 * sy)
    gx = (float(img[s, y, min(X-1, x+2)]) - float(img[s, y, max(0, x-2)])) / (4 * sx)
    seed = np.array([gz, gy, gx], float)
    if np.linalg.norm(seed) < 1e-3 or (abs(gz) + abs(gy) + abs(gx)) < 20:
        bone = img[s] > 300
        if np.count_nonzero(bone) > 20:
            ys, xs = np.where(bone)
            seed = np.array([0.0, (y - float(ys.mean())) / sy, (x - float(xs.mean())) / sx])
        if np.linalg.norm(seed) < 1e-3:
            seed = np.array([0.0, 0.0, 1.0])
    n0 = seed / np.linalg.norm(seed)

    # 2) 좁은 원뿔에서 최단 골통과 방향 선택
    ds = min(sx, sy, 0.5) or 0.5
    N = int(round(18.0 / ds))
    t = np.arange(-N, N + 1).astype(float) * ds

    a = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(a, n0))) > 0.9:
        a = np.array([0.0, 1.0, 0.0])
    u = a - np.dot(a, n0) * n0; u = u / np.linalg.norm(u)
    v = np.cross(n0, u); v = v / np.linalg.norm(v)

    if has_scipy:
        tilts = [-24, -16, -8, 0, 8, 16, 24]
        dirs = []
        for ta in tilts:
            for tb in tilts:
                d = n0 + np.tan(np.radians(ta)) * u + np.tan(np.radians(tb)) * v
                nn = np.linalg.norm(d)
                if nn > 1e-6:
                    dirs.append(d / nn)
        dirs = np.array(dirs)
    else:
        dirs = n0[None, :]

    K = dirs.shape[0]; M = t.shape[0]
    zc = (s + np.outer(dirs[:, 0] / sz, t)).ravel()
    yc = (y + np.outer(dirs[:, 1] / sy, t)).ravel()
    xc = (x + np.outer(dirs[:, 2] / sx, t)).ravel()
    coords = np.vstack([zc, yc, xc])
    if has_scipy:
        vals = map_coordinates(img, coords, order=1, mode='nearest', output=np.float32)
    else:
        zz = np.clip(np.round(zc), 0, Z-1).astype(int)
        yy = np.clip(np.round(yc), 0, Y-1).astype(int)
        xx = np.clip(np.round(xc), 0, X-1).astype(int)
        vals = img[zz, yy, xx].astype(np.float32)
    profs = vals.reshape(K, M)

    if has_scipy and M >= 7:
        try:
            pss = savgol_filter(profs, 7, 2, axis=1)
        except Exception:
            pss = profs
    else:
        pss = profs
    mask = pss > 150
    has = mask.any(axis=1)
    first = np.argmax(mask, axis=1)
    last = M - 1 - np.argmax(mask[:, ::-1], axis=1)
    span = np.where(has, last - first, 10**9)
    contains = has & (first - 8 <= N) & (N <= last + 8)
    span = np.where(contains, span, 10**9)
    kb = int(np.argmin(span))
    if span[kb] >= 10**9:
        return None

    ps = pss[kb]; bs = int(first[kb]); be = int(last[kb]); n = dirs[kb]
    bp = ps[bs:be + 1]
    if len(bp) < 4:
        return None

    n3 = max(1, len(bp) // 3)
    outer_hu = float(np.max(bp[:n3]))
    inner_hu = float(np.max(bp[-n3:])) if len(bp[-n3:]) > 0 else outer_hu
    midseg = bp[n3:-n3] if len(bp) > 2 * n3 else bp
    diploe_hu = float(np.min(midseg)) if len(midseg) > 0 else float(np.mean(bp))
    cortex_hu = (outer_hu + inner_hu) / 2.0
    if cortex_hu <= 0:
        return None
    sdr = diploe_hu / cortex_hu
    if sdr > 1.5 or sdr < 0.05:
        return None

    outer_thr = max(otsu * 0.5, outer_hu * 0.5, 150)
    inner_thr = max(otsu * 0.5, inner_hu * 0.5, 150)
    seg = ps[bs:be + 1]
    ao = np.where(seg >= outer_thr)[0]
    ai = np.where(seg >= inner_thr)[0]
    mo = bs + int(ao[0]) if len(ao) else bs
    mi = bs + int(ai[-1]) if len(ai) else be
    if mi <= mo:
        mo, mi = bs, be

    span_len = mi - mo
    if span_len >= 3:
        third = max(1, span_len // 3)
        po = mo + int(np.argmax(ps[mo:mo + third + 1]))
        pi = mi - int(np.argmax(ps[mi - third:mi + 1][::-1]))
        # 피질판-해면 경계 = 피질피크와 해면바닥의 '중점'(절반강하점).
        # 0.5*peak(half-max)는 고밀도 해면을 피질로 오인 → 해면 과소측정. 중점으로 보정.
        if span_len >= 6:
            dfloor = float(np.min(ps[mo + span_len // 4: mi - span_len // 4 + 1]))
        else:
            dfloor = float(np.min(ps[mo:mi + 1]))
        ho = 0.5 * (float(ps[po]) + dfloor); hi = 0.5 * (float(ps[pi]) + dfloor)
        od = mo + span_len // 2
        for i in range(po, mi + 1):
            if ps[i] < ho:
                od = i; break
        idd = mo + span_len // 2
        for i in range(pi, mo - 1, -1):
            if ps[i] < hi:
                idd = i; break
        if od > idd:
            od = idd = mo + int(np.argmin(ps[mo:mi + 1]))
        outer_mm = round(max(0, od - mo) * ds, 2)
        inner_mm = round(max(0, mi - idd) * ds, 2)
        diploe_mm = round(max(0, idd - od) * ds, 2)
    else:
        outer_mm = inner_mm = diploe_mm = 0.0
    total_mm = round(span_len * ds, 2)
    oblique_deg = round(float(np.degrees(np.arcsin(min(1.0, abs(float(n[0])))))), 1)

    return {
        'sdr': round(sdr, 3),
        'cortex_hu': round(cortex_hu, 1),
        'outer_hu': round(outer_hu, 1),
        'inner_hu': round(inner_hu, 1),
        'diploe_hu': round(diploe_hu, 1),
        'outer_mm': outer_mm,
        'diploe_mm': diploe_mm,
        'inner_mm': inner_mm,
        'total_mm': total_mm,
        'oblique_deg': oblique_deg,
    }
