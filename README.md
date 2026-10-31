# Kernel-paired calvarial CT measurement pipeline

Code and de-identified derived data accompanying:

> Nam TM, Park Y, Yang NR, Kim YG. *Metric-specific effects of the reconstruction kernel on
> automated calvarial CT measurements: a paired agreement study of 111 navigation CT
> examinations.* Frontiers in Neuroscience (under review).

The pipeline measures calvarial thickness and attenuation metrics on paired bone-kernel and
brain-kernel reconstructions of the same head CT acquisition. Localization geometry is derived
from the bone reconstruction and applied unchanged to the brain reconstruction, so that only
the reconstruction kernel differs between the two measurements.

## Repository layout

```
pipeline/   measurement code (runs on DICOM; DICOM data are not distributed)
analysis/   statistics that reproduce the article's tables from the derived data
data/       de-identified per-examination derived measurements (111 examinations)
```

### pipeline/

| File | Role |
|---|---|
| `spherical_sdr_module.py` | Spherical atlas (8 hemispheric regions); voxel-based regional SDR, cortical and diploic attenuation; 40 fixed angular targets; surface-normal profile measurement (`measure_normal_layers`) giving outer, diploic, inner and total thickness and table-peak attenuation |
| `sdr_io.py` | DICOM series loading (SimpleITK), Otsu threshold on bone voxels (300–2,000 HU), DICOM metadata, and per-reconstruction export (`save_to_excel`), including regional aggregation of fixed-point metrics with the obliquity belt (targets with normal obliquity > 22°, parietal > 60°, excluded). Extracted unchanged from the authors' interactive viewer, which is not distributed. |
| `batch_pair_run.py` | Runs the bone/brain pair for every examination. Detects the external-auditory-canal level, skull center, radius and z-scaling on the bone series and applies them to the brain series. `--otsu own` (default; each series its own Otsu threshold) or `--otsu shared` (bone threshold applied to the brain series). Writes `PairDB_<mode>.csv/.xlsx` and `PairDB_<mode>_meta.csv`. |
| `aggregate_pairdb.py` | Converts `PairDB_<mode>.csv` (one row per reconstruction) into the per-examination dataset: global value = mean of all successfully measured hemispheric regions; regional value = mean of the available sides; exclusions (age < 18, flagged device, repeat examinations). Verified to reproduce the published dataset. |

### analysis/

`reproduce_tables.py` recomputes Table 2 (automated vs manual thickness), Table 3 (agreement,
all metrics), Table 4 (stratified agreement and scanner–kernel-specific mappings), Supplementary
Table S1 (shared threshold), S2 (SDR category discordance) and S6 (stratified agreement and
between-pair comparisons) from the files in `data/`.

```bash
pip install -r requirements.txt
python3 analysis/reproduce_tables.py --out results
```

Bootstrap confidence intervals depend on the random seed (`--seed`); with 5,000 resamples they
match the published values to the reported precision, although the last digit can differ.

### data/

| File | Content |
|---|---|
| `derived_measurements_own_threshold.csv` | Primary analysis: each reconstruction thresholded independently. Suffix `_bone` = sharp kernel, `_brain` = smooth kernel. |
| `derived_measurements_shared_threshold.csv` | Sensitivity analysis: bone-series Otsu threshold applied to the brain series. |
| `validation_subset.csv` | Automated vs blinded manual total thickness at one low-obliquity target in 19 examinations. |

Examinations are identified only by study codes (FIN-001 to FIN-111). Ages of 90 years or older
are recorded as 90 to reduce re-identification risk (one examination). Units: thickness in mm,
attenuation in HU, SDR dimensionless. Blank cells denote values that could not be measured.

## Running the pipeline on your own data

Input layout:

```
<input_root>/
  <examination_folder>/
    <series>_bone_setting/        DICOM files, sharp (bone) kernel, 2-mm axial
    <series>_non_bone_setting/    DICOM files, smooth (brain) kernel, same acquisition
```

```bash
python3 pipeline/batch_pair_run.py <input_root>                 # own threshold (primary analysis)
python3 pipeline/batch_pair_run.py <input_root> --otsu shared   # shared-threshold sensitivity analysis
python3 pipeline/aggregate_pairdb.py <input_root>/PairDB_own.csv codemap.csv out_own.csv --exclude-id <id>
```

`codemap.csv` has two columns, `folder,code`, mapping each examination folder to a study code.
**`PairDB_*` output contains DICOM identifiers (name, ID, birth date); keep it and the code map
private.** Both are excluded by `.gitignore`.

## Notes

* Profile-based metrics (thickness, table peak) fail in a hemispheric region when no surface-normal
  direction crosses bone at the target, the bone segment is < 2 mm, or the profile ratio is outside
  0.05–1.5; voxel-based metrics require ≥ 50 bone voxels per region.
* The research skull density ratio is defined by intensity partitioning and is not the value
  reported by any treatment system.
* Full parameter values are given in the Methods section of the article.
* Source DICOM images cannot be shared: the institutional review board approval
  (Ewha Womans University Mokdong Hospital, EUMC 2026-08-020) and Korean privacy legislation do not
  permit redistribution of patient images.

## License and citation

Code is released under the MIT License (`LICENSE`). Please cite the article and this repository
(`CITATION.cff`); the archived version has a Zenodo DOI.
