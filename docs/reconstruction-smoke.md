# Retrospective reconstruction smoke check

This bounded engineering check accepts explicit checkpoint and prepared-image paths. It records strict EMA/raw loading, checkpoint/source hashes, source dirty state, case identifiers and subject-key membership, acquisition settings, seeds, metrics and raw arrays. Outputs must be new directories. It does not modify input images or checkpoint files.

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python scripts/reconstruct_smoke.py --checkpoint /path/to/last.ckpt \
  --h5 /path/to/complex-t1.h5 --training-h5 /path/to/training-store.h5 \
  --output results/smoke-local
```

The default finite-Langevin DAPS uses a unitary masked Fourier acquisition with synthetic noise specified per real component. Zero-filled, ridge/CG and denoising controls accompany multiple DAPS draws on a common magnitude display scale. `--inner-kernel gaussian` is diagnostic only; a correct local Gaussian solve can yield poor outer reconstruction. See [sampler conventions](samplers.md).

For the PTB Slurm cluster, create `logs/` and submit `sbatch scripts/smoke.slurm` with the same checkpoint/HDF5 arguments. The recipe requests one H100 for ten minutes; adjust scheduler directives for another cluster.

`--require-held-out` rejects detected subject-key overlap and missing membership checks. Different identifiers may still represent the same patient, so this is not patient-independence verification. Retrospective cases can overlap training; two-draw spread is not calibrated uncertainty. This check validates neither acquired multicoil operators nor nonlinear joint quantitative inversion. A scientific benchmark needs a documented patient-level split, multiple cases/draws and prespecified metrics.

The historical branch sweep and machine-specific notebook extraction tools remain in the forensic archive; this PR publishes the reusable explicit-input smoke tool only.
