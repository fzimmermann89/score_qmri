# Plan: Cleanup and JEPA CLI Integration

## Tasks

### 1. Rename `lightning_module.py` → `diffusion_module.py`

**Files to modify:**
- `src/score_qmri/training/lightning_module.py` → Rename to `diffusion_module.py`
- `src/score_qmri/training/__init__.py` → Update import path
- `src/score_qmri/cli.py` → Update import path

### 2. Add JEPA to CLI

**Approach:** Create a separate CLI entry point for JEPA training (`score-qmri-jepa`).

**Why separate?** 
- JEPA and Diffusion have different purposes (pretraining vs generation)
- Different hyperparameters and workflows
- Cleaner separation of concerns

**Files to create/modify:**
- Create `src/score_qmri/jepa_cli.py` - New CLI for JEPA training
- Update `pyproject.toml` - Add `score-qmri-jepa` console script

**CLI structure:**
```bash
# Diffusion training (existing)
score-qmri-train fit --config diffusion_config.yaml

# JEPA pretraining (new)
score-qmri-jepa fit --config jepa_config.yaml
```

### 3. Update README

**Add JEPA section:**
- Explain JEPA pretraining purpose
- Show example config
- Document CLI usage

**Updated structure:**
```markdown
# Score-Based Posterior Sampling for Quantitative MRI

## Highlights
- VE-SDE (EDM/Karras) diffusion with dual x₀ heads
- **JEPA pretraining** for self-supervised MRI representation learning
- ...

## Quick Start

### Diffusion Training
```bash
score-qmri-train fit --config config.yaml
```

### JEPA Pretraining
```bash
score-qmri-jepa fit --config jepa_config.yaml
```

## JEPA Config Example
```yaml
model:
  img_size: 256
  patch_size: 16
  embed_dim: 384
  encoder_depth: 12
  ...
```
```

### 4. Update `training/__init__.py`

Export both modules:
```python
from .diffusion_module import DiffusionLightningModule
from .jepa_module import JEPALightningModule
```

## Summary of Changes

| File | Action |
|------|--------|
| `training/lightning_module.py` | Rename → `diffusion_module.py` |
| `training/__init__.py` | Update imports, add JEPA |
| `cli.py` | Update import path |
| `jepa_cli.py` | **NEW** - JEPA CLI entry point |
| `pyproject.toml` | Add `score-qmri-jepa` script |
| `README.md` | Add JEPA documentation |

## Open Questions

None - the plan is straightforward.
