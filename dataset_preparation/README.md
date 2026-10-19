# Dataset preparation

These standalone recipes recover the preparation code for **all 69 distinct HDF5 filenames** in `data.yaml` after the WAND T1-map correction in PR #27. The old YAML references 68 filenames because it uses `wand_t1w.h5` for two different kinds. This folder does not change training, existing datasets, or the older scripts in `scripts/`.

Start from the original downloaded NIfTI, DICOM, MATLAB, NPZ or k-space files. Every command uses explicit input/output paths and refuses existing output files, including symlinks. No preparation, model loading or downloading happens at import. CPU is the default; reconstruction and generation require an explicit device to use CUDA. Failed conversions leave their new partial outputs for inspection; they never delete or resume files automatically.

**Coverage is not a claim that every historical voxel has been reproduced.** The catalog records validation separately for every output. Most checks use one complete representative subject per file. A real M4Raw GRE reconstruction still differs from the selected historical file. IBC T1/M0 and complex reconstruction have numerical discrepancies. Historical MONAI samples cannot be regenerated from a fresh random stream because the original seed was not recorded.

## Environment

Run commands from the repository root with Python 3.12. Keep this environment separate from environments used by training:

```bash
uv venv --python 3.12 .prep-venv
source .prep-venv/bin/activate
# CPU wheels avoid installing CUDA libraries on a preparation-only machine.
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install -r dataset_preparation/requirements.txt pytest
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The numerical checks used the exact versions listed in `requirements.txt`, with the existing Torch 2.9.1 installation. Backend, BLAS and device changes can affect fitted/complex values. The optional MONAI generator additionally needs a compatible MONAI installation and the complete local bundle; normal preparation, repacking and tests do not need MONAI. No complete MONAI inference was run during this audit, so generation compatibility is not certified.

## Find a recipe for each HDF5

```bash
python -m dataset_preparation.catalog
python -m dataset_preparation.catalog --yaml data.yaml
python -m dataset_preparation.catalog wand_t1map.h5
```

The first command lists every filename and its validation status. The last prints the exact command and its limitations. `catalog.json` includes the reference file SHA-256, source identities, companion outputs, and command arguments for every file. The stored HDF5 hashes identify the audited references; regenerated HDF5 containers may have different bytes despite identical arrays.

To run one bounded recipe, provide the **raw family input root** and a **new output directory**:

```bash
python -m dataset_preparation.catalog ixi_t1w.h5 \
  --input-root /path/to/downloaded/ixi --output-dir /path/to/new/ixi --limit 1 --execute
python -m dataset_preparation.catalog quiero_t1map.h5 \
  --input-root /path/to/quiero/4Pub --output-dir /path/to/new/quiero --limit 1 --execute
```

`--limit 1` is a subject or session limit, not necessarily a single HDF5 key. For Deli-CS it limits planes: use `--limit 360` to recreate the complete ordered set; the command builds all three output maps together. IBC commands build paired parameter/S0 files together. Run each such command once, rather than invoking it again for its companion filename. Use each module's `--help` for full-build and subject-selection options. No batch command silently starts all 69 recipes.

## Original inputs and script families

- `openneuro.py`: ds000221 (six outputs), ds002685 (three), ds002785, ds002790, ds003967, ds004611 (four), and ds004889 (two). Inputs are downloaded dataset roots, including the actual git-annex content. Preserve native precision: some ds000221 magnitude outputs are float64. Historical IPython state supplies the ds000221 T1 mask and the missing ds002785/FLAIR producers.
- `mrf.py`: four Zenodo 8183344 outputs from `MR?` sessions with original/reposition NIfTI files; three NIST outputs from `MRF_invivo_anon` DICOM series. NIST deliberately preserves the original first-five-file selection before slice-position sorting. Filesystem order is not portable: pass `--dicom-order-manifest` with a JSON mapping of series keys to relative DICOM filenames when rebuilding from newly extracted files.
- `nifti.py`: IXI T1/T2/PD; Quiero `4Pub` three-parameter `*test.nii`; ds005851 MNI R1/R2/PD derivatives; Rockland anatomy. The selected Rockland recipe is the executed earlier history revision, not the last exploratory saved script. ds005851 retains the source's rate conventions and zero-phase complex64 storage for `_m0_abs`.
- `radunsky.py`: `mrQ_maps.mat`, `T2map_EMC.mat` and `T2_star_map.mat` in subject directories. T2 and T2* are converted from milliseconds to seconds.
- `delics_maps.py`: the 360 `qmaps_vol*_slice*.npz` files under `Deli-CS/qmaps`. The bundled `delics_order.json` records the historical plane order. All 360 planes and all three arrays were compared exactly. Sorting filenames would change slice order.
- `wand.py`: WAND sourcedata NIfTI files and the **complete four-table lookup table** for MP2RAGE. Use `wand_lut.py` to generate the historical UK7T simulation table explicitly, or provide the trusted original `LUT_UK7T.pkl`. The older `lut_uk7t.npz` with only PSIR/MP2RAGE/T1 is incomplete for M0 fitting. The historical non-inplace clamp is intentionally preserved.
- `ibc_relaxo.py`: IBC Relaxometry sourcedata. Recovered executed T1/M0 uses B1/1000, not the later saved /800 experiment. T2*/S0 uses the original 3D mask and `[20:-20]` crop. The commented T2 writer was restored from the paired fit and checked against the target.
- `ulf.py`: subject `3T` folders under the original `ulfenc/train3` input. Despite the historical `ulf_` filenames, these H5 files contain **3T reference anatomy**, not the separate Enhanced low-field predictions.
- `brainweb.py`: cached original raw-short gzip classes or the newer flat `sXX.h5` subject files. Explicit little-endian decoding is validated. `pack` preserves the newer arrays; `raw` reconstructs them a slice at a time. The historical unsigned subtraction mode preserves the chosen source; `--normalization float-safe` deliberately changes a few known out-of-training-range voxels in subject04. Never reinterpret the stored uint8 H5 as a byte-order repair.
- `raw_mri.py`: fastMRI brain multicoil raw H5, M4Raw repeats, and CC k-space. CPU FFT/ESPIRiT/SVD may be expensive even for one subject. `--limit 1`, one thread and explicit output paths are defaults. `fastmri-prepared` accepts the archived pre-curation fastMRI H5 directory; `cc-prepared` accepts the archived pre-curation `cc.h5`; `m4raw-prepared` accepts a directory of the four archived pre-curation M4Raw H5 files. These reuse explicit intermediates and reproduce the selected slice-filtered arrays exactly in the checked subjects. Raw alternatives remain separately runnable with the catalog’s `--raw` option. Raw `cc-raw` and the older `cc-fix` conversion are distinct source variants; neither is certified as the selected corpus producer.
- `latent_t1.py`: `repack` copies saved `samples_500.h5` arrays and attributes to `ldm_t1w.h5`. `generate` runs an already downloaded MONAI brain synthesis bundle using an explicit new seed and device. Bundle revision from the original record: `55d88321e08930ba3a71d084ed94afaab0b29320`. Repacking is reproducible; new generation is not a reproduction of the historical draws.

## WAND lookup table

Generating the complete historical LUT is a deliberate, potentially hour-long CPU job. Reuse an existing complete table where possible:

```bash
python -m dataset_preparation.wand_lut --output /path/to/new/uk7t_complete.npz
python -m dataset_preparation.wand --input-root /path/to/WAND \
  --output-dir /path/to/new/wand --modalities t1map t1w m0_complex m0_mag \
  --lut /path/to/new/uk7t_complete.npz --limit 1
```

To reuse your trusted historical pickle, pass `--lut /path/to/LUT_UK7T.pkl --trust-pickle-lut` directly to `wand`. The catalog expects a complete NPZ and does not opt into pickle loading.

## Cleanup and shape normalization

`clean_h5.py` faithfully retains slices with at least 1000 pixels above absolute magnitude 1e-6. For grouped complex data the combined group supplies the common indices. `normalize_h5.py` preserves the eleven historical padding/cropping and excluded-subject recipes. Both stream slices to exclusive new outputs and reproduce the historical loss of HDF5 attributes.

**Do not apply either stage to the entire collection.** Later regenerations bypassed them: ds004889 FLAIR intentionally keeps 90 padded slices; ds002685 FLAIR and ds004611 keep native spatial dimensions. The catalog describes the selected revision. Quiero and Zenodo default to applying their verified final slice filter in memory before writing; `--stage prepared` exposes the original pre-curation arrays without creating an extra whole-dataset copy.

For an explicitly identified archived intermediate:

```bash
python -m dataset_preparation.clean_h5 --help
python -m dataset_preparation.normalize_h5 --help
```

HDF5 compression/chunks are recorded separately from logical arrays. Brainweb uses the same training schema but new compression chunks. Do not require container MD5 equality as a substitute for comparing scientific contents.

## Downloads

Preparation reads existing downloads. `download.py` is a separate explicit action:

- `openneuro`: requires a dataset ID, pinned revision and new clone path; only `--get` paths fetch annex content. DataLad/git-annex must be installed separately. A plain Git clone contains links, not necessarily images.
- `rockland`: accepts the original `aws_links.csv`, explicit subjects and modalities. It prints a plan unless `--download` is supplied. Each file has a required byte budget and downloads use anonymous public S3 HTTPS.
- `archive`: downloads a supplied HTTPS URL into a new file, optionally checks SHA-256, and requires a byte budget. It does not extract, delete or authenticate. Obtain current archive URLs and permissions from the dataset provider.
- Brainweb raw mode downloads missing classes only with its explicit `--download` flag.

Expired/private fastMRI signed URLs and authentication material are excluded. The source stash contained these URLs; copying them into a public PR would not make a usable downloader. Original raw access agreements remain applicable.

## Validation and known limits

`validation.json` summarizes the bounded real-data checks and `catalog.json` gives per-file status. Exact full-subject comparisons cover all 25 OpenNeuro/MRF files, IXI, Quiero, ds005851, Radunsky, Rockland, ULF and WAND T1/T1w/native magnitude. Deli-CS has a complete 360-plane comparison for every map. Brainweb has earlier complete logical packing verification and fresh raw checks on nine subject04 planes. Saved LDM samples and prepared CC, all three fastMRI modalities, and all four M4Raw modalities also have exact checked reproduction paths. These archived-intermediate paths are distinguished from raw reconstruction.

The following remain explicitly weaker:

- WAND M0: maximum complex difference 7.75e-7 and magnitude difference 4.77e-7 in one full subject.
- IBC T2/S0: maximum differences 5.96e-8 seconds and 1.67e-6. T2*/S0 is exact in one full subject.
- IBC T1/M0: new code matches execution of the recovered original functions exactly, but differs from the historical file by up to 0.001164 seconds / 0.02322 respectively. The cause is unresolved; do not silently choose loose tolerances.
- fastMRI T1: one full real case has matching support, combined maximum error 1.36e-5 and coilwise maximum 7.07e-4. T2/FLAIR end-to-end checks were not run.
- M4Raw GRE: meaningful mismatch remains (mean absolute error about 0.0041, maximum about 0.488). The cleaned code and the recovered original current script give bit-identical CPU outputs on that actual input; neither reproduces the historical archive. Source ordering and old/new calibration variants were investigated; the cause remains unresolved. Other M4Raw modalities and raw CC have not been regenerated end-to-end.
- Full MONAI generation and complete fresh-source downloads were not run. Missing historical RNG state prevents exact regeneration of synthetic samples. The archived-sample repack remains an exact logical reproduction path.

The final [M4Raw forensic report](validation/M4RAW-FINAL-FORENSICS.md) records the executed-source, raw-copy and numerical probes. The actual January 5 executed producer and helpers were recovered; independent Echo/Ceph raw file hashes match, and the historical padding implementation also matches. No evidence establishes a GPU/backend explanation for the remaining mismatch.

These limits are not hidden by the filename coverage check. The PR makes all recipes inspectable and runnable, but does not certify every historical dataset as reproducible from raw input yet.

Verify a newly prepared file against the selected reference without loading whole volumes:

```bash
python -m dataset_preparation.verify_h5 --help
python -m pytest --noconftest dataset_preparation/tests -q
```

The verifier defaults to every value, exact comparison, and independent shape/dtype/schema checks. Optional attribute/layout checks and explicit tolerances are separate; byte-exact and numerical-equivalence fields remain distinct. Explicit sampling is labeled as sampling. Source SHA-256 values and the execution environment are preserved in `provenance.json`. Real checks used low-priority CPU work, at most one thread per process, no scheduler/GPU jobs and roughly 150 MB of new audit artifacts rather than duplicating the full collection.
