# Dataset preparation

Cleaned copies of the original `datacheck` and IPython preparation scripts. Masks, units, orientation, normalization and crops retain the recovered behavior. Commands read existing downloads and create new H5 files; existing outputs are refused. Run from the repository root with Python 3.12.

```bash
uv venv --python 3.12 .prep-venv
source .prep-venv/bin/activate
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r dataset_preparation/requirements.txt pytest
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m dataset_preparation.openneuro --dataset ds000221 \
  --input-root /path/to/ds000221 --output-dir /path/to/new/ds000221 --limit 1
python -m dataset_preparation.nifti --dataset ixi --modalities t1w \
  --input-root /path/to/IXI --output-dir /path/to/new/ixi --limit 1
```

The scripts below take `--input-root`, `--output-dir`, optional `--subjects`, `--limit` unless noted. `--modalities` selects outputs and is required for `nifti` and `wand`. Use `python -m dataset_preparation.SCRIPT --help` for arguments. A limit selects subjects/sessions; Deli-CS counts planes and raw MRI counts subject/modality pairs. Raw MRI and IBC default to one; use `--all` for a full build. Other scripts process all selected inputs when the limit is omitted. Computation uses CPU by default.

## Outputs and original sources

The filename patterns below cover all 69 outputs after the WAND T1-map fix in #27. Braces list the available modalities. IBC writes two paired outputs per command; Deli-CS writes all three maps together.

| Script and selector | Output filenames | Original source / expected input |
| --- | --- | --- |
| `openneuro --dataset ds000221` | `ds000221_{gre,t1w,t1map,mprage,t2w,flair}.h5` | `v3.py` and executed IPython205 masks; downloaded dataset root |
| `openneuro --dataset ds002685` | `ds002685_{t1w,t2w,flair}.h5` | `v4.py`; downloaded dataset root |
| `openneuro --dataset ds002785` | `ds002785_t1w.h5` | IPython211:30/77; downloaded dataset root |
| `openneuro --dataset ds002790` | `ds002790_t1w.h5` | `v5.py`; downloaded dataset root |
| `openneuro --dataset ds003967` | `ds003967_t1map_cropped.h5` | `v6.py`; downloaded dataset root |
| `openneuro --dataset ds004611` | `ds004611_{t1map,t2map,t1w,mprage}.h5` | `v7.py`; downloaded dataset root |
| `openneuro --dataset ds004889` | `ds004889_{t1w,flair}.h5` | `v9.py`, IPython328:64; downloaded dataset root |
| `nifti --dataset ixi` | `ixi_{t1w,t2w,pd}.h5` | `ixi.py`, IPython298:63 for T2/PD; IXI NIfTI directory |
| `nifti --dataset quiero` | `quiero_{t1map,t2map,m0map}.h5` | `v12.py`; `4Pub` with three-channel `*test.nii` files |
| `nifti --dataset ds005851` | `ds005851_{t1map,t2map,m0_abs}.h5` | `ds5851.py`; MNI R1/R2/PD derivatives |
| `nifti --dataset rockland` | `rockland_{t1w,t2w,flair}.h5` | IPython343:404/409/411; downloaded BIDS anatomy |
| `mrf --dataset zenodo8183344` | `mrf_zenodo8183344_{t1map,t2map,t1w,t2w}.h5` | `v10.py`; MR? folders with original/reposition NIfTIs |
| `mrf --dataset nist` | `mrf_nist_{t1map,t2map,m0map}.h5` | `v11.py`, IPython215:156/163; `MRF_invivo_anon` DICOM series |
| `radunsky` | `radunsky_{t1,m0mag,t2,t2star}.h5` | `radunsky.py`; subject folders with `mrQ_maps.mat`, `T2map_EMC.mat`, `T2_star_map.mat` |
| `delics_maps` | `delics_{t1map,t2map,complexm0}.h5` | `v15.py`; `Deli-CS/qmaps` NPZ files |
| `wand` | `wand_{t1map,t1w,m0_complex,m0_mag,t2w,t1w_neu}.h5` | `wand2.py`; WAND sourcedata and complete MP2RAGE LUT |
| `ibc_relaxo --family t1` | `ibc_relaxo_{t1map,m0map}.h5` | IPython278:16/19/22; IBC sourcedata |
| `ibc_relaxo --family t2star` | `ibc_relaxo_{t2starmap,s0map}.h5` | IPython224:6/32/40; IBC sourcedata |
| `ibc_relaxo --family t2` | `ibc_relaxo_{t2map,s0map_t2}.h5` | `v13.py`; IBC sourcedata |
| `ulf` | `ulf_{t1,t2,flair}.h5` | HPC IPython49:60/61/62; `ulfenc/train3/subject/3T` references |
| `raw_mri fastmri` | `fastmri_{t1,t2,flair}.h5` | `fastmri.py`; brain multicoil k-space H5 files |
| `raw_mri m4raw` | `m4raw_{t1w,t2w,flair,gre}.h5` | `m4raw.py` and editor history `hR8v.py`; raw repeat H5 files |
| `raw_mri cc-raw` | `cc_reconstructed.h5` (unverified candidate for `cc.h5`) | `v16.py` cc_6 recipe; CC k-space H5 files |
| `brainweb pack` / `brainweb raw` | `brainweb.h5` | `mr2/phantoms/brainweb.py`; newer `sXX.h5` files or raw-short gzip classes |
| `latent_t1 repack` / `latent_t1 generate` | `ldm_t1w.h5` | `monai/run.py`; saved samples or the local MONAI bundle |

OpenNeuro inputs need the actual git-annex contents, not just a Git clone. NIST retains the original first five files before sorting by slice location; filesystem order may differ after extraction. `--dicom-order-manifest` can supply the original filenames. Deli-CS uses the included `delics_order.json` for the verified 360-plane order; omit `--limit` to build all planes.

IBC takes `--input` and `--output` directories:

```bash
python -m dataset_preparation.ibc_relaxo --family t1 \
  --input /path/to/ibc/sourcedata --output /path/to/new/ibc --limit 1
```

WAND MP2RAGE needs all four lookup tables. The older three-key `lut_uk7t.npz` is incomplete for M0 fitting. Reuse a complete NPZ or your original `LUT_UK7T.pkl` (with `--trust-pickle-lut`). Generating a full table takes about one CPU hour:

```bash
python -m dataset_preparation.wand_lut --output /path/to/new/uk7t_complete.npz
python -m dataset_preparation.wand --input-root /path/to/WAND \
  --output-dir /path/to/new/wand --modalities t1map t1w m0_complex m0_mag \
  --lut /path/to/new/uk7t_complete.npz --limit 1
```

Brainweb and synthetic T1 use a single `--output` file:

```bash
python -m dataset_preparation.brainweb pack --input-root /path/to/newer/brainweb \
  --output /path/to/new/brainweb.h5 --limit 1
python -m dataset_preparation.brainweb raw --input-root /path/to/gzip-cache \
  --output /path/to/new/brainweb.h5 --subjects 04
python -m dataset_preparation.latent_t1 repack --input-root /path/to/samples_500.h5 \
  --output /path/to/new/ldm_t1w.h5 --limit 1
python -m dataset_preparation.latent_t1 generate --bundle-root /path/to/local/monai-bundle \
  --output /path/to/new/ldm_t1w.h5 --limit 1 --seed 0 --device cpu
```

Brainweb raw classes are cached as `sXX/CLASS.gz`; add `--download` to fetch missing classes from Brainweb. Little-endian decoding and the original unsigned baseline subtraction reproduce the selected newer version. Synthetic generation requires compatible MONAI and bundle revision `55d88321e08930ba3a71d084ed94afaab0b29320`; it was not run in full. The original seed was not saved, so only repacking reproduces the historical samples exactly.

## Existing prepared H5 files

`clean_h5.py` applies the original slice filter (at least 1000 pixels above absolute magnitude 1e-6) and drops attributes. Combined images select slices for both complex groups. It reproduces checked archived CC, fastMRI and M4Raw intermediates without reconstructing raw k-space:

```bash
python -m dataset_preparation.clean_h5 --input /path/to/prepared/cc.h5 \
  --output /path/to/new/cc.h5 --limit 1
```

`normalize_h5.py` retains the eleven original shape/subject-exclusion recipes from `scripts/preprocess_h5.py`. `raw_mri cc-fix` retains the separate legacy `fix_cc.py` axis correction; use it only with that old CC layout. Do not apply either shape normalization or cleanup indiscriminately: ds004889 FLAIR keeps 90 padded slices, and ds002685 FLAIR/ds004611 retain their native widths. Quiero and Zenodo already include their selected final slice filter; `--stage prepared` exposes their earlier arrays.

## Downloads

`download.py openneuro` uses DataLad/git-annex (installed separately); provide `--dataset`, `--revision`, a new `--output` and explicit relative `--get` paths. `download.py rockland` uses the original `aws_links.csv`, `--manifest`, `--subjects`, `--modalities` and a new `--output`. Other recipes read existing downloads. Expired/private fastMRI signed URLs are not included.

## Reproduction checks

Representative full subjects matched exactly across OpenNeuro/MRF, IXI, Quiero, ds005851, Radunsky, Rockland, ULF and WAND T1/magnitude outputs. All 360 Deli-CS planes matched; Brainweb packing was checked across all 20 subjects, with fresh raw checks on nine planes. These are bounded checks, not a fresh full-corpus rebuild.

Remaining differences: WAND M0 is within 7.75e-7; IBC T2/S0 within 5.96e-8 seconds / 1.67e-6. IBC T1/M0 matches execution of the recovered functions but differs from historical H5 by up to 0.001164 seconds / 0.02322. fastMRI T1 raw combined/coilwise maxima are 1.36e-5 / 7.07e-4; other raw contrasts are unchecked. M4Raw GRE still has a meaningful raw mismatch (MAE about 0.0041, maximum 0.488), despite matching execution of the recovered producer. Raw CC is unverified. Those discrepancies remain unresolved; archived-intermediate cleanup is a separate verified path.

```bash
python -m pytest --noconftest dataset_preparation/tests -q
```
