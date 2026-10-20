# Dataset preparation

Independent scripts for each dataset family, cleaned from the original `datacheck` and IPython code. Each script contains its own helpers and can run without importing another script. The header states its outputs, required inputs and where to download them. Masks, units, orientation, normalization and crops retain the recovered behavior. Commands read existing downloads and create new H5 files; existing outputs are refused. Use Python 3.12; the examples below run from the repository root.

```bash
uv venv --python 3.12 .prep-venv
source .prep-venv/bin/activate
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r dataset_preparation/requirements.txt pytest
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python dataset_preparation/ds000221.py \
  --input-root /path/to/ds000221 --output-dir /path/to/new/ds000221 --limit 1
python dataset_preparation/ixi.py --modalities t1w \
  --input-root /path/to/IXI --output-dir /path/to/new/ixi --limit 1
```

Most scripts take `--input-root`, `--output-dir`, `--subjects` and `--limit`. `--modalities` selects outputs where a dataset has several contrasts (required for WAND). Run `python dataset_preparation/SCRIPT.py --help` for arguments. A limit selects eligible subjects/sessions per output; Deli-CS counts planes and raw MRI counts subject/modality pairs. CC, fastMRI, M4Raw and IBC default to one; use `--all` for a full build. Other scripts process all selected inputs when the limit is omitted. Computation uses CPU by default.

## Outputs and original sources

The filename patterns below cover all 69 outputs after the WAND T1-map fix in #27. Braces list the available modalities. IBC writes two paired outputs per command; Deli-CS writes all three maps together.

| Script and selector | Output filenames | Original source / expected input |
| --- | --- | --- |
| `ds000221.py` | `ds000221_{gre,t1w,t1map,mprage,t2w,flair}.h5` | `v3.py` and executed IPython205 masks; downloaded dataset root |
| `ds002685.py` | `ds002685_{t1w,t2w,flair}.h5` | `v4.py`; downloaded dataset root |
| `ds002785.py` | `ds002785_t1w.h5` | IPython211:30/77; downloaded dataset root |
| `ds002790.py` | `ds002790_t1w.h5` | `v5.py`; downloaded dataset root |
| `ds003967.py` | `ds003967_t1map_cropped.h5` | `v6.py`; downloaded dataset root |
| `ds004611.py` | `ds004611_{t1map,t2map,t1w,mprage}.h5` | `v7.py`; downloaded dataset root |
| `ds004889.py` | `ds004889_{t1w,flair}.h5` | `v9.py`, IPython328:64; downloaded dataset root |
| `ixi.py` | `ixi_{t1w,t2w,pd}.h5` | `ixi.py`, IPython298:63 for T2/PD; IXI NIfTI directory |
| `quiero.py` | `quiero_{t1map,t2map,m0map}.h5` | `v12.py`; `4Pub` with three-channel `*test.nii` files |
| `ds005851.py` | `ds005851_{t1map,t2map,m0_abs}.h5` | `ds5851.py`; MNI R1/R2/PD derivatives |
| `rockland.py` | `rockland_{t1w,t2w,flair}.h5` | IPython343:404/409/411; downloaded BIDS anatomy |
| `mrf_zenodo8183344.py` | `mrf_zenodo8183344_{t1map,t2map,t1w,t2w}.h5` | `v10.py`; MR? folders with original/reposition NIfTIs |
| `mrf_nist.py` | `mrf_nist_{t1map,t2map,m0map}.h5` | `v11.py`, IPython215:156/163; `MRF_invivo_anon` DICOM series |
| `radunsky.py` | `radunsky_{t1,m0mag,t2,t2star}.h5` | `radunsky.py`; subject folders with `mrQ_maps.mat`, `T2map_EMC.mat`, `T2_star_map.mat` |
| `delics_maps.py` | `delics_{t1map,t2map,complexm0}.h5` | `v15.py`; `Deli-CS/qmaps` NPZ files |
| `wand.py` | `wand_{t1map,t1w,m0_complex,m0_mag,t2w,t1w_neu}.h5` | `wand2.py`; WAND sourcedata and complete MP2RAGE LUT |
| `ibc_t1.py` | `ibc_relaxo_{t1map,m0map}.h5` | IPython278:16/19/22; IBC sourcedata |
| `ibc_t2star.py` | `ibc_relaxo_{t2starmap,s0map}.h5` | IPython224:6/32/40; IBC sourcedata |
| `ibc_t2.py` | `ibc_relaxo_{t2map,s0map_t2}.h5` | `v13.py`; IBC sourcedata |
| `ulf.py` | `ulf_{t1,t2,flair}.h5` | HPC IPython49:60/61/62; `ulfenc/train3/subject/3T` references |
| `fastmri.py` | `fastmri_{t1,t2,flair}.h5` | `fastmri.py`; brain multicoil k-space H5 files |
| `m4raw.py` | `m4raw_{t1w,t2w,flair,gre}.h5` | `m4raw.py` and editor history `hR8v.py`; raw repeat H5 files |
| `cc.py` | `cc.h5` (prepared input); `--raw` writes unverified `cc_reconstructed.h5` | `v16.py` cc_6 recipe; CC k-space H5 files |
| `brainweb.py pack` / `brainweb.py raw` | `brainweb.h5` | `mr2/phantoms/brainweb.py`; newer `sXX.h5` files or raw-short gzip classes |
| `latent_t1.py repack` / `latent_t1.py generate` | `ldm_t1w.h5` | `monai/run.py`; saved samples or the local MONAI bundle |

OpenNeuro inputs need the actual git-annex contents, not just a Git clone. Retain the clone layout for ds002785, ds002790 and ds003967: the historical H5 keys include annex object filenames. NIST retains the original first five files before sorting by slice location; filesystem order may differ after extraction. `--dicom-order-manifest` can supply the original filenames. Deli-CS uses the included `delics_order.json` for the verified 360-plane order; omit `--limit` to build all planes.

IBC fits take sourcedata and output directories:

```bash
python dataset_preparation/ibc_t1.py \
  --input-root /path/to/ibc/sourcedata --output-dir /path/to/new/ibc --limit 1
```

WAND MP2RAGE needs all four lookup tables. The older three-key `lut_uk7t.npz` is incomplete for M0 fitting. Reuse a complete NPZ or your original `LUT_UK7T.pkl` (with `--trust-pickle-lut`). Generating a full table takes about one CPU hour:

```bash
python dataset_preparation/wand_lut.py --output /path/to/new/uk7t_complete.npz
python dataset_preparation/wand.py --input-root /path/to/WAND \
  --output-dir /path/to/new/wand --modalities t1map t1w m0_complex m0_mag \
  --lut /path/to/new/uk7t_complete.npz --limit 1
```

Brainweb and synthetic T1 use a single `--output` file:

```bash
python dataset_preparation/brainweb.py pack --input-root /path/to/newer/brainweb \
  --output /path/to/new/brainweb.h5 --limit 1
python dataset_preparation/brainweb.py raw --input-root /path/to/gzip-cache \
  --output /path/to/new/brainweb.h5 --subjects 04
python dataset_preparation/latent_t1.py repack --input-root /path/to/samples_500.h5 \
  --output /path/to/new/ldm_t1w.h5 --limit 1
python dataset_preparation/latent_t1.py generate --bundle-root /path/to/local/monai-bundle \
  --output /path/to/new/ldm_t1w.h5 --limit 1 --seed 0 --device cpu
```

Brainweb raw classes are cached as `sXX/CLASS.gz`; add `--download` to fetch missing classes from Brainweb. Little-endian decoding and the original unsigned baseline subtraction reproduce the selected newer version. Synthetic generation requires compatible MONAI and bundle revision `55d88321e08930ba3a71d084ed94afaab0b29320`; it was not run in full. The original seed was not saved, so only repacking reproduces the historical samples exactly.

## Matching archived prepared H5 files

`clean_h5.py` applies the original slice filter (at least 1000 pixels above absolute magnitude 1e-6) and drops attributes. Combined images select slices for both complex groups. It reproduces checked archived CC, fastMRI and M4Raw intermediates without reconstructing raw k-space. `cc.py` uses this input by default; `fastmri.py --prepared` and `m4raw.py --prepared` read the corresponding archived files from `--input-root`:

```bash
python dataset_preparation/clean_h5.py --input /path/to/prepared/cc.h5 \
  --output /path/to/new/cc.h5 --limit 1
```

`normalize_h5.py` retains the eleven original shape/subject-exclusion recipes from `scripts/preprocess_h5.py`. `cc.py --legacy` retains the separate legacy `fix_cc.py` axis correction; use it only with that old CC layout. Do not apply either shape normalization or cleanup indiscriminately: ds004889 FLAIR keeps 90 padded slices, and ds002685 FLAIR/ds004611 retain their native widths. Quiero, Zenodo and ds002785 already include their selected final slice filter.

## Downloads

`download.py openneuro` uses DataLad/git-annex (installed separately); provide `--dataset`, `--revision`, a new `--output` and explicit relative `--get` paths. `download.py rockland` uses the original `aws_links.csv`, `--manifest`, `--subjects`, `--modalities` and a new `--output`. Other recipes read existing downloads. Expired/private fastMRI signed URLs are not included. The Radunsky paper’s repository link returned 404 during this check; the original saved session archives remain usable.

## Reproduction checks

Fresh comparisons against `score_qmri_consistent_2026-10-19_le` cover all 69 filenames: 63 have exact arrays, shapes and dtypes in the checked scope. Most checks use one complete representative subject per file, including every coil for grouped complex data. Deli-CS covers all 360 planes; Brainweb covers all 362 raw planes of subject04 (625,604,056 values), alongside the earlier 20-subject packing check. Synthetic T1 covers one complete saved sample. These checks do not constitute a full-corpus rebuild.

Six outputs retain numerical differences in the checked subjects; maximum absolute differences are:

- `wand_m0_complex.h5`: 7.75e-7.
- `wand_m0_mag.h5`: 4.77e-7.
- `ibc_relaxo_t1map.h5`: 0.001164 seconds.
- `ibc_relaxo_m0map.h5`: 0.02322.
- `ibc_relaxo_t2map.h5`: 5.96e-8 seconds.
- `ibc_relaxo_s0map_t2.h5`: 1.67e-6.

The recovered IBC T1/M0 functions reproduce execution of the original functions exactly; their drift from historical H5 is unresolved. WAND T1/T1w and IBC T2*/S0 match their checked subjects exactly.

The exact CC/fastMRI/M4Raw comparisons use archived prepared H5 inputs followed by the original slice filter. Raw fastMRI T1 has combined/coilwise maximum differences of 1.36e-5 / 7.07e-4; other raw contrasts are unchecked. Raw M4Raw GRE retains a meaningful mismatch (MAE about 0.0041, maximum 0.488), despite matching execution of the recovered producer. Raw CC is unverified. The missing original RNG seed prevents regeneration of the historical synthetic draws. No tolerances or reference-derived corrections conceal these limits.

```bash
python -m pytest --noconftest dataset_preparation/tests -q
```
