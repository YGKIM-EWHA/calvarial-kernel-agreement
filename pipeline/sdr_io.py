"""
sdr_io.py — DICOM loading, Otsu thresholding and per-examination export used by the
paired-kernel analysis. Extracted verbatim from sdr_viewer_v2.py (the interactive Flask
viewer) so that the analysis can run without the web interface.

Functions kept exactly as used by batch_pair_run.py:
  load_dicom(dicom_dir)        -> reads a DICOM series with SimpleITK, computes the Otsu
                                  threshold on bone voxels (300–2,000 HU), stores image,
                                  spacing and DICOM metadata in the module-level `current` dict
  save_to_excel(...)           -> aggregates the eight hemispheric atlas regions (voxel metrics)
                                  and the fixed-point profile metrics (obliquity belt: >22°
                                  excluded, parietal >60°) and appends one row per reconstruction
                                  to EXCEL_PATH (.csv, plus a formatted .xlsx when openpyxl is available)
`EXCEL_PATH` is set by the caller before use.
"""
import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_otsu
import os, csv
from datetime import datetime
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

current = {'img': None, 'otsu': None, 'total': 0, 'spacing': None, 'path': None}
EXCEL_PATH = os.path.join(os.getcwd(), 'SDR_Database.xlsx')

def save_to_excel(patient_id, path, otsu, eac_slice, regional_sdr, fixed_points):
    """환자 SDR 데이터를 Excel에 누적 저장 (csv 방식으로 안정적 저장)"""
    import csv
    meta = current.get('meta', {})

    atlas_regions = ['전두골_좌','전두골_우','두정골_좌','두정골_우',
                     '측두골_좌','측두골_우','후두골_좌','후두골_우']

    # 헤더 정의
    headers = [
        'No','Patient_Name','Patient_ID','Birth_Date','Age','Sex',
        'Study_Date','Study_Time','Study_Description',
        'Modality','Institution','Manufacturer','Scanner_Model',
        'Weight_kg','Height_m','KVp','Slice_Thickness_mm',
        'DICOM_Path','Total_Slices','Otsu_HU','EAC_Slice'
    ]
    for r in atlas_regions:
        en = r.replace('전두골','Frontal').replace('두정골','Parietal').replace('측두골','Temporal').replace('후두골','Occipital').replace('_좌','_L').replace('_우','_R')
        headers += [f'Atlas_{en}_SDR', f'Atlas_{en}_Cortex_HU', f'Atlas_{en}_Diploe_HU',
                    f'Atlas_{en}_Outer_mm', f'Atlas_{en}_Diploe_mm', f'Atlas_{en}_Inner_mm',
                    f'Atlas_{en}_Total_mm', f'Atlas_{en}_CortexPeak_HU']
    for region, points in fixed_points.items():
        for pt in points:
            headers.append(f'FP_{pt["name"]}_SDR')
        # 부위별 평균 컬럼
        en_r = region.replace('전두골','Frontal').replace('두정골','Parietal').replace('측두골','Temporal').replace('후두골','Occipital').replace('_좌','_L').replace('_우','_R')
        headers.append(f'FP_{en_r}_MEAN')

    # CSV 경로
    csv_path = EXCEL_PATH.replace('.xlsx', '.csv')

    # 기존 파일에서 행 번호 파악
    next_no = 1
    file_exists = os.path.exists(csv_path)
    if file_exists:
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                rows = list(csv.reader(f))
                next_no = len(rows)  # 헤더 포함이므로 len = 마지막 번호 + 1
        except:
            file_exists = False
            next_no = 1

    # 데이터 행 구성
    row_data = [
        next_no,
        meta.get('patient_name', patient_id),
        meta.get('patient_id', ''),
        meta.get('birth_date', ''),
        meta.get('age', ''),
        meta.get('sex', ''),
        meta.get('study_date', datetime.now().strftime('%Y-%m-%d')),
        meta.get('study_time', ''),
        meta.get('study_desc', ''),
        meta.get('modality', 'CT'),
        meta.get('institution', ''),
        meta.get('manufacturer', ''),
        meta.get('model', ''),
        meta.get('weight_kg', ''),
        meta.get('height_m', ''),
        meta.get('kvp', ''),
        meta.get('slice_thickness', ''),
        path,
        current.get('total', ''),
        round(otsu, 1),
        eac_slice,
    ]
    def _region_layer_means(region_key):
        """해당 부위 고정점들의 층 두께/피질피크 평균.
        belt: 측방 부위는 oblique>22° 제외(축상에서 신뢰).
              두정부(vertex)는 법선이 본래 craniocaudal이라 3D 법선/MPR로 측정 →
              oblique>60°만 제외(거의 순수 수직, 2mm z해상도로 신뢰불가)."""
        thr = 60 if str(region_key).startswith('두정골') else 22
        pts = fixed_points.get(region_key, []) or []
        acc = {'outer_mm':[], 'diploe_mm':[], 'inner_mm':[], 'total_mm':[], 'cortex_peak_hu':[]}
        for pt in pts:
            ob = pt.get('oblique_deg')
            if ob is not None and ob > thr:   # belt
                continue
            for k in acc:
                vv = pt.get(k)
                if vv is not None and vv != '':
                    acc[k].append(float(vv))
        out = {}
        for k, arr in acc.items():
            out[k] = round(sum(arr)/len(arr), 2) if arr else ''
        return out

    for r in atlas_regions:
        d = regional_sdr.get(r, {})
        row_data.append(d.get('sdr') or '')
        row_data.append(d.get('cortex_hu') or '')
        row_data.append(d.get('diploe_hu') or '')
        lm = _region_layer_means(r)
        row_data.append(lm['outer_mm'])
        row_data.append(lm['diploe_mm'])
        row_data.append(lm['inner_mm'])
        row_data.append(lm['total_mm'])
        row_data.append(lm['cortex_peak_hu'])
    for region, points in fixed_points.items():
        valid = []
        for pt in points:
            sv = pt.get('sdr')
            row_data.append(sv or '')
            if sv: valid.append(sv)
        # 부위별 평균
        row_data.append(round(sum(valid)/len(valid), 3) if valid else '')

    # unicode surrogate 제거 함수
    def safe_str(v):
        if v is None or v == '': return ''
        try:
            s = str(v)
            return s.encode('utf-8', errors='replace').decode('utf-8')
        except: return ''

    # row_data 전체 안전 변환
    safe_row = []
    for v in row_data:
        if isinstance(v, (int, float)):
            safe_row.append(v)
        else:
            safe_row.append(safe_str(v))

    # CSV 저장
    with open(csv_path, 'a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(safe_row)

    # CSV → Excel 변환 (openpyxl 사용 가능하면)
    if HAS_OPENPYXL:
        try:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = 'SDR_Database'

            thin = Side(style='thin', color='DDDDDD')
            border = Border(left=thin, right=thin, top=thin, bottom=thin)
            center = Alignment(horizontal='center', vertical='center')

            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                rows = list(csv.reader(f))

            for r_idx, row in enumerate(rows, 1):
                for c_idx, val in enumerate(row, 1):
                    cell = ws.cell(row=r_idx, column=c_idx)
                    # 숫자 변환
                    try:
                        if '.' in str(val): cell.value = float(val)
                        elif str(val).isdigit(): cell.value = int(val)
                        else: cell.value = val
                    except: cell.value = val
                    cell.border = border
                    cell.alignment = center
                    cell.font = Font(name='Arial', size=9)

                    # 헤더 스타일
                    if r_idx == 1:
                        if 'Atlas_' in str(val):
                            cell.fill = PatternFill(start_color='1F4E79', end_color='1F4E79', fill_type='solid')
                        elif 'FP_' in str(val):
                            cell.fill = PatternFill(start_color='375623', end_color='375623', fill_type='solid')
                        else:
                            cell.fill = PatternFill(start_color='2E75B6', end_color='2E75B6', fill_type='solid')
                        cell.font = Font(name='Arial', bold=True, color='FFFFFF', size=9)

                    # SDR 색상
                    h = rows[0][c_idx-1] if rows else ''
                    if ('_SDR' in h or '_MEAN' in h) and r_idx > 1:
                        try:
                            v = float(val)
                            if v >= 0.4: color='C6EFCE'
                            elif v >= 0.3: color='FFEB9C'
                            else: color='FFC7CE'
                            cell.fill = PatternFill(start_color=color, end_color=color, fill_type='solid')
                        except: pass

            # 열 너비
            col_w = [6,20,16,14,8,8,14,10,24,10,24,20,20,10,10,8,12,44,10,10,10]
            for i, w in enumerate(col_w, 1):
                ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
            for i in range(len(col_w)+1, len(rows[0])+1 if rows else 1):
                ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = 11
            ws.row_dimensions[1].height = 40
            ws.freeze_panes = 'H2'

            wb.save(EXCEL_PATH)
            return True, EXCEL_PATH
        except Exception as e:
            return True, csv_path  # CSV라도 성공

    return True, csv_path



def load_dicom(dicom_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        return False, "DICOM 시리즈 없음"
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    img = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    skull_all = img[(img > 300) & (img < 2000)]
    if len(skull_all) < 100:
        return False, "두개골 데이터 부족"
    otsu = threshold_otsu(skull_all)
    current['img'] = img
    current['otsu'] = float(otsu)
    current['total'] = img.shape[0]
    current['spacing'] = spacing

    # ── DICOM 메타데이터 추출 ──
    def get_meta(key):
        try:
            v = reader.GetMetaData(0, key).strip()
            if not v: return ''
            return v.encode('utf-8', errors='replace').decode('utf-8')
        except: return ''

    # 이름: DICOM은 "성^이름" 형식
    raw_name = get_meta('0010|0010')
    patient_name = raw_name.replace('^', ' ').strip()

    # 생년월일 → 나이 계산
    dob = get_meta('0010|0030')      # YYYYMMDD
    study_date = get_meta('0008|0020')  # YYYYMMDD
    age_val = get_meta('0010|1010')  # 직접 나이 태그 (예: "065Y")
    age = ''
    if age_val:
        age = ''.join(filter(str.isdigit, age_val))
    elif dob and len(dob) == 8 and study_date and len(study_date) == 8:
        try:
            from datetime import date
            bd = date(int(dob[:4]), int(dob[4:6]), int(dob[6:8]))
            sd = date(int(study_date[:4]), int(study_date[4:6]), int(study_date[6:8]))
            age = str((sd - bd).days // 365)
        except: age = ''

    # 검사일 포맷 변환 YYYYMMDD → YYYY-MM-DD
    def fmt_date(d):
        if d and len(d) >= 8:
            try: return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            except: return d
        return d

    # 체중/키 단위 정리
    def fmt_float(v):
        try: return round(float(v), 1)
        except: return v

    meta = {
        'patient_name':  patient_name,
        'patient_id':    get_meta('0010|0020'),
        'birth_date':    fmt_date(dob),
        'age':           age,
        'sex':           get_meta('0010|0040'),       # M / F
        'study_date':    fmt_date(study_date),
        'study_time':    get_meta('0008|0030')[:6],   # HHMMSS
        'study_desc':    get_meta('0008|1030'),
        'modality':      get_meta('0008|0060'),
        'institution':   get_meta('0008|0080'),
        'weight_kg':     fmt_float(get_meta('0010|1030')),
        'height_m':      fmt_float(get_meta('0010|1020')),
        'manufacturer':  get_meta('0008|0070'),
        'model':         get_meta('0008|1090'),
        'kvp':           get_meta('0018|0060'),
        'slice_thickness': fmt_float(get_meta('0050|0018') or get_meta('0018|0050')),
    }
    current['meta'] = meta
    return True, "OK"

