# Dataset scripts

Prepare the datasets used by `data.yaml`. Each script's header lists the required inputs and download links.

| Script | Output |
| --- | --- |
| [brainweb.py](brainweb.py) | `brainweb.h5` |
| [cc.py](cc.py) | `cc.h5` |
| [delics_maps.py](delics_maps.py) | `delics_{t1map,t2map,complexm0}.h5` |
| [download.py](download.py) | OpenNeuro or NKI-Rockland source files |
| [ds000221.py](ds000221.py) | `ds000221_{gre,t1w,t1map,mprage,t2w,flair}.h5` |
| [ds002685.py](ds002685.py) | `ds002685_{t1w,t2w,flair}.h5` |
| [ds002785.py](ds002785.py) | `ds002785_t1w.h5` |
| [ds002790.py](ds002790.py) | `ds002790_t1w.h5` |
| [ds003967.py](ds003967.py) | `ds003967_t1map_cropped.h5` |
| [ds004611.py](ds004611.py) | `ds004611_{t1map,t2map,t1w,mprage}.h5` |
| [ds004889.py](ds004889.py) | `ds004889_{t1w,flair}.h5` |
| [ds005851.py](ds005851.py) | `ds005851_{t1map,t2map,m0_abs}.h5` |
| [fastmri.py](fastmri.py) | `fastmri_{t1,t2,flair}.h5` |
| [ibc_t1.py](ibc_t1.py) | `ibc_relaxo_{t1map,m0map}.h5` |
| [ibc_t2.py](ibc_t2.py) | `ibc_relaxo_{t2map,s0map_t2}.h5` |
| [ibc_t2star.py](ibc_t2star.py) | `ibc_relaxo_{t2starmap,s0map}.h5` |
| [ixi.py](ixi.py) | `ixi_{t1w,t2w,pd}.h5` |
| [latent_t1.py](latent_t1.py) | `ldm_t1w.h5` |
| [m4raw.py](m4raw.py) | `m4raw_{t1w,t2w,flair,gre}.h5` |
| [mrf_nist.py](mrf_nist.py) | `mrf_nist_{t1map,t2map,m0map}.h5` |
| [mrf_zenodo8183344.py](mrf_zenodo8183344.py) | `mrf_zenodo8183344_{t1map,t2map,t1w,t2w}.h5` |
| [quiero.py](quiero.py) | `quiero_{t1map,t2map,m0map}.h5` |
| [radunsky.py](radunsky.py) | `radunsky_{t1,m0mag,t2,t2star}.h5` |
| [rockland.py](rockland.py) | `rockland_{t1w,t2w,flair}.h5` |
| [ulf.py](ulf.py) | `ulf_{t1,t2,flair}.h5` |
| [wand.py](wand.py) | `wand_{t1map,t1w,m0_complex,m0_mag,t2w,t1w_neu}.h5` |
| [wand_lut.py](wand_lut.py) | WAND MP2RAGE lookup table (NPZ, `--output`) |

For example, from the repository root:

```bash
python dataset_preparation/ds000221.py --input-root RAW --output-dir NEW
```

Commands process all matching inputs. Run a script with `--help` for its options. Scripts refuse existing output files.

CC, fastMRI and M4Raw reconstruct raw k-space and apply the final slice filter in the same run. For complex data, the combined image selects slices for every coil. Quiero, Zenodo MRF and ds002785 also include their final slice filters.

Raw CC remains unverified against the historical H5 files; raw M4Raw differs from them. WAND M0 and IBC fits retain numerical differences. Reuse saved synthetic samples to reproduce the historical synthetic T1 data.
