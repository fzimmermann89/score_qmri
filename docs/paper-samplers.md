# Paper-form posterior samplers with EDM denoisers

EDM preconditioning and denoising training are compatible with DPS and DAPS.
These classes preserve the algorithms' stochastic transitions and conditioning
rules while using the existing clean-image denoiser. They return **approximate
posterior samples**: matching either paper does not establish calibrated
uncertainty for a learned MRI prior.

## Ancestral DPS

`AncestralDPSSampler` implements the ancestral DDPM proposal followed by the
DPS correction through the denoiser Jacobian. For an EDM state `u` at noise
level `sigma`, define `a = 1/(1+sigma**2)` and VP state `z = sqrt(a)*u`.
The clean estimate is `D(z/sqrt(a), sigma)`. The equivalent epsilon estimate
is `(z-sqrt(a)*D)/sqrt(1-a)`. Neither conversion requires retraining.

For a transition to `sigma_next`, the ancestral proposal in EDM coordinates has
mean `r*u + (1-r)*D` and variance `sigma_next**2*(1-r)`, where
`r = sigma_next**2/sigma**2`. Convert that proposal back to VP coordinates
before subtracting the conditioning gradient **with respect to z**.

The default `conditioning="residual_norm"` matches the official DPS Gaussian
conditioning code: differentiate the measurement residual norm and multiply
by an explicit guidance scale. `conditioning="gaussian_likelihood"` instead
uses the summed Gaussian negative log likelihood and requires `tau`.
These choices are exposed separately because their gradients differ.

The implementation uses fixed-small DDPM posterior variance. The reference
implementation also supports learned variance; the EDM denoiser has no learned
variance head. Clipping is disabled by default and can be enabled explicitly.
The Karras noise grid stays within the checkpoint's supported noise range;
it need not reproduce a particular VP training beta schedule.

```python
from score_qmri.sampling import AncestralDPSConfig, AncestralDPSSampler

sampler = AncestralDPSSampler(model, AncestralDPSConfig(
    num_steps=1000, guidance_scale=0.1,
))
draw = sampler.sample(operator, measurement, kind, generator=generator)
```

## DAPS

`PaperDAPSSampler` implements the paper's PF-ODE endpoint, conditional Langevin
updates, and independent re-noising. The default Gaussian approximation centers
the conditional prior at the PF-ODE endpoint with radius `sigma`.
`prior="score"` selects the alternative conditional prior score from equation 6,
evaluated using a configurable positive low-noise prior score.

Langevin noise has its full `sqrt(2*step_size)` amplitude. There is no implicit
temperature reduction, clipping, step cap, final projection or extra denoising.
The Euler PF-ODE stops at `sigma_x0`, as in the reference. The outer schedule
and linear learning-rate decay reproduce the official scheduler, including its
omitted positive minimum and final zero. Noise levels above the checkpoint's
supported maximum are rejected instead of silently extrapolating the prior.

`tau` is the measurement noise standard deviation **per real coordinate** in
the same units as `operator(draw)`. For complex noise, its expected squared
magnitude is `2*tau**2`. The Gaussian energy is `sum(abs(residual)**2)/(2*tau**2)`.
The official DAPS unadjusted Langevin code differentiates a full squared loss
and divides by `tau**2`, giving twice this drift. Reference parity therefore
uses `tau_reference=sqrt(2)*tau`; this implementation keeps the paper's Gaussian
likelihood convention.

```python
from score_qmri.sampling import PaperDAPSConfig, PaperDAPSSampler

sampler = PaperDAPSSampler(model, PaperDAPSConfig(
    tau=0.01, anneal_steps=200, reverse_steps=5,
    langevin_steps=100, langevin_lr=1.5e-5,
))
draw = sampler.sample(operator, measurement, kind, generator=generator)
```

Step sizes must suit the operator and noise units. Reducing `tau` increases
likelihood curvature and can require a smaller explicit Langevin step size.
The defaults are a starting recipe, not settings validated for every acquisition.

## Verification and interpretation

Independent comparisons executed the frozen official source against the same
Gaussian denoiser, operator, initial state and random streams. DPS matched all
16 reverse states to a maximum absolute error of `1.28e-7` for both clean-image
and epsilon adapters. DAPS matched the official Gaussian and score-prior modes
to `1.52e-8`, after the explicit likelihood convention conversion above.
Unit tests independently check DDPM coefficients, both conditional prior
gradients, stochasticity, reproducibility, schedule details and model limits.

To repeat reference parity, place the official source trees at
`references/DAPS` and `references/diffusion-posterior-sampling`, checked out at
the commits linked below, and install the reference-only dependency
`torchdiffeq==0.2.5`. Run:

```bash
python scripts/check_reference_samplers.py --references references --output reference-parity.json
```

The script verifies the used reference files against a pinned SHA-256 manifest
before executing their declarations. It omits unrelated heavy imports; the
transition, conditioning and scheduler declarations remain unmodified.

Existing `DPSSampler` and `DAPSSampler` remain available as historical
reconstruction variants. Their additional stabilization rules are not used by
these new classes. In particular, the new DPS has no RMS gradient normalization
or inverse-sigma guidance multiplier.

DPS uses a plug-in likelihood approximation. Practical DAPS uses an approximate
conditional prior and finitely many unadjusted Langevin steps. Learned-score
error, discretization and mixing remain relevant even with faithful code.
Evaluate posterior moments and coverage on known-posterior controls as well as
image quality; PSNR, measurement agreement and different random seeds alone
cannot establish calibrated posterior sampling.

The analytic Gaussian control in `scripts/check_posterior_moments.py` exposes
this distinction directly. With prior variance `0.04` and measurement noise
`tau=0.01`, the observed posterior variance is `9.975e-5`; the null-space
variance remains `0.04`. In 4096 draws, default norm-conditioned ancestral DPS
gave about 15.3 times the observed variance despite matching the reference.
Default DAPS's decaying learning rate also mixed poorly at the late stages.
Keeping the step size constant (`lr_min_ratio=1`) corrected that mixing failure
in this control. The `daps-refined` score-prior preset, with 16 PF steps,
300 inner steps and learning rate `5e-6`, gave variance ratios of 1.017/1.065
in the observed real/imaginary coordinates and 1.060/1.017 in the null space.
Monte Carlo relative standard error for each variance estimate is about 2.2%.
These are measured approximation errors, not a universal calibration guarantee.

```bash
python scripts/check_posterior_moments.py --preset dps --output dps-moments.json
python scripts/check_posterior_moments.py --preset daps-default --output daps-default-moments.json
python scripts/check_posterior_moments.py --preset daps-refined --output daps-refined-moments.json
```

More inner steps alone do not fix the PF-centered Gaussian conditional prior
approximation. The score-prior option instead targets the product of the
measurement likelihood, `p(xt|x0)`, and the low-noise prior, at the cost of one
score-model evaluation per inner step. Convergence checks should vary step size,
inner mixing effort and outer noise levels separately. The finite low-noise
prior remains a smoothed approximation to the clean prior.

`scripts/validate_paper_samplers.py` records seeded reconstructions, configuration,
checkpoint provenance, source hashes and metrics for an explicitly supplied
frozen Fourier case. It makes no held-out or posterior-calibration claim.

## Ensembles, uncertainty and reconstruction errors

`scripts/sample_posterior_ensemble.py` generates independent batched draws for
one fixed measurement. The default DPS recipe uses 1000 ancestral steps and
norm-guidance scale .1. The default DAPS recipe uses the dynamic score prior,
32 outer stages, 5 PF steps, 100 Langevin steps, constant learning rate 1.5e-5,
and prior score at sigma=.01. The measurement likelihood uses tau=.01 per real
coordinate. These explicit engineering-screen settings can be overridden.

```bash
python scripts/sample_posterior_ensemble.py \
    --checkpoint model.ckpt --cases cases.npz --case-index 0 \
    --kind COMPLEX_T1W_COIL_COMBINED --sampler dps --output ensemble-dps
python scripts/sample_posterior_ensemble.py \
    --checkpoint model.ckpt --cases cases.npz --case-index 0 \
    --kind COMPLEX_T1W_COIL_COMBINED --sampler daps --output ensemble-daps
python scripts/analyze_posterior_ensembles.py \
    --runs ensemble-dps ensemble-daps --output uncertainty-comparison
```

Each output directory must be new. The runner saves eight draws by default,
four per batch. Within-batch random tensors have independent entries; separate
batches use separate seeds. A batch seed is recorded with its draw indices.
Changing batch size changes the random-number stream and therefore the draws.
All draws start independently from the prescribed Gaussian initialization;
this is an ensemble across complete sampler trajectories, not adjacent MCMC
states from one trajectory. More draws reduce Monte Carlo estimation noise,
but do not fix algorithm or learned-prior bias.

The analysis makes two matched comparisons:

- Magnitude: mean `E[|x|]`, absolute error relative to `|truth|`, and sample SD
  of `|x|`. This mean is different from `|E[x]|`.
- Complex: mean `E[x]`, error `|E[x]-truth|`, and complex spread
  `sqrt(Var(real)+Var(imag))`. This exposes phase uncertainty that magnitude
  maps alone can hide. The displayed mean image is `|E[x]|`.

All variances use the n-1 sample denominator. SD describes variation across
draws; it is not divided by sqrt(n), which would estimate uncertainty in the
Monte Carlo mean. Error and SD maps share the same full-range color limits
across compared samplers within a case/domain. No smoothing, spatial filtering
or ROI masking is applied to the displayed maps. Additional focused PNG assets
use a shared limit of 10% of the reference peak, with all saturation fractions
reported; the full-range maps remain available. PNG/PDF figures and numeric
maps are saved alongside raw draw hashes and a JSON summary.

Spatial error–spread diagnostics use a fixed foreground rule, reference
magnitude above 5% of its peak, to prevent background from dominating. They
include RMS error versus RMS SD, rank correlation and eight equal-count bins
ordered by uncertainty. Pixels are correlated and this is one fixed noisy
measurement; these diagnostics are not coverage or posterior-calibration tests.
Acquired and missing Fourier locations are also summarized separately, using
complex error against the noiseless reference and complex ensemble spread. This
can expose suppressed measurement-space uncertainty that image-space totals hide.
The first four draws, 2/4/full prefix diagnostics and split-half mean differences
help reveal Monte Carlo variation. Eight draws provide an exploratory uncertainty map, not a precise
tail-probability estimate.

References:

- [DPS paper](https://arxiv.org/abs/2209.14687),
  [official implementation](https://github.com/DPS2022/diffusion-posterior-sampling/tree/effbde7325b22ce8dc3e2c06c160c021e743a12d).
- [DAPS paper](https://arxiv.org/abs/2407.01521),
  [official implementation](https://github.com/zhangbingliang2019/DAPS/tree/0b182726f84df1ad26fde4dccebefda4747b82d8).
