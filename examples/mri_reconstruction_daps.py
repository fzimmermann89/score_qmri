# %% [markdown]
# # DAPS MRI Reconstruction Example
#
# This example demonstrates Decoupled Annealing Posterior Sampling (DAPS) for MRI reconstruction
# using the package sampler with mrpro's Fourier operator and FastMRI phantom data.
#
# DAPS combines an unconditional diffusion prior with likelihood guidance to solve
# the inverse problem: k-space → image.
#
# Requires: `pip install -e ".[examples]"` (mrpro) and a FastMRI H5 file.

# ruff: noqa: E402
# %%

from pathlib import Path

import mrpro
import torch
from mrpro.data import CsmData, SpatialDimension
from mrpro.operators import FourierOp
from mrpro.phantoms.fastmri import FastMRIKDataDataset

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.daps_sampler import DAPSConfig, DAPSSampler

# %% Configuration
CHECKPOINT_PATH = Path("checkpoints/last.converted.ckpt")
FASTMRI_H5 = ["file_brain_AXT1_206_2060051.h5"]
BATCH = 1
SAMPLE = 4
# %% Set up the problem
k = FastMRIKDataDataset(FASTMRI_H5)[SAMPLE]
# Fix up data to match the model input format
k.header.recon_matrix = SpatialDimension(1, 256, 256)
k.traj.kx, k.traj.ky = k.traj.ky, k.traj.kx.flip(-1)
k.header.encoding_matrix.x, k.header.encoding_matrix.y = (k.header.encoding_matrix.y, k.header.encoding_matrix.x)
k.header.encoding_fov.x, k.header.encoding_fov.y = (k.header.encoding_fov.y, k.header.encoding_fov.x)
k.data *= 3e3

# Ground truth

csm = CsmData.from_kdata_walsh(k)
x_gt = mrpro.algorithms.reconstruction.DirectReconstruction(k, csm=csm)(k).data
# Retrospective undersampling
k_us = k[..., ::4, :]
k_us = k_us.cuda()
csm = csm.cuda()
fourier_op = FourierOp.from_kdata(k_us)
op = fourier_op @ csm.as_operator()

# %% Load the model
model = DiffusionModel.from_checkpoint(CHECKPOINT_PATH, compile_model=False).cuda().eval()

# %%
# Direct (zero-filled) reconstruction for reference.
(x_zf,) = op.H(k_us.data)

# CG reconstruction for reference.
(x_cg,) = mrpro.algorithms.optimizers.cg(op.gram + 0.01, x_zf)


# Set tau from the measured noise after applying the k-space scaling above.
cfg = DAPSConfig()

# %%
sampler = DAPSSampler(model, cfg)

y = k_us.data.repeat_interleave(BATCH, dim=0)

with torch.no_grad():
    (recon,) = sampler.sample(
        op,
        y,
        noise_std=0.03,
        template=x_zf.repeat_interleave(BATCH, dim=0),
        kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED,
        noise=None,
    )
# %%
import matplotlib.pyplot as plt


def plot(img: torch.Tensor, ax: plt.Axes, title: str) -> None:
    img = img.detach().abs().squeeze().cpu()
    vmax = img[img >= 0.01 * img.max()].quantile(0.9).item()
    im = ax.imshow(img, cmap="gray", vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.axis("off")


fig, ax = plt.subplots(1, 5 if BATCH > 1 else 4, figsize=(30, 6))
plot(x_gt, ax[0], "Ground truth")
plot(x_zf, ax[1], "Zero-filled")
plot(x_cg, ax[2], "CG")
plot(recon.mean(0), ax[3], "DAPS")
if BATCH > 1:
    plot(recon.std(0), ax[4], "Sample spread (uncalibrated)")
fig.tight_layout()

# %%
