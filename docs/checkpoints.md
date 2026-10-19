# Checkpoint compatibility and training launch

Load a trusted checkpoint with its recorded architecture and an explicit evaluation weight selection:

```python
from score_qmri.models.diffusion import DiffusionModel

model = DiffusionModel.from_checkpoint(
    "last.ckpt", weights="ema", compile_model=False, map_location="cpu"
)
print(model.checkpoint_info)
```

The loader reconstructs current recorded UNet specifications and older flat native-UNet fields, maps compiled wrapper keys to eager names, and loads strictly by default. It refuses to guess missing architecture. `weights="raw"` selects training weights; `"auto"` retains the earlier EMA-if-present behavior. Historical VP/cosine checkpoints require their compatible historical model code.

New checkpoints serialize configuration as plain data, preserving integers and booleans and excluding runtime logger/callback objects. Optimizer, scheduler, callback and EMA training states remain separate and intact.

To convert an existing trusted checkpoint to portable metadata:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python scripts/convert_checkpoint_format.py last.ckpt --output last.portable.ckpt
```

Conversion refuses to overwrite the input or an existing output. Legacy input uses `weights_only=False`, so only load trusted checkpoints. Conversion preserves saved training tensors; it does not recreate missing dataloader state or prove deterministic mid-epoch DDP continuation.

Launch training with the current split configuration:

```bash
score-qmri fit --config data.yaml --config model.yaml --job_name my-run
```

`train.slurm` uses the same two files. Resume still requires the appropriate explicit training configuration and storage layout. Checkpoint filenames identify saved training steps, rather than a nonexistent `train_loss` metric.
