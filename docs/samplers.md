# Posterior sampler conventions and limits

This page describes the historical `DPSSampler` and `DAPSSampler` variants.
For the ancestral DPS and paper-form DAPS implementations, including how an
EDM denoiser maps into VP coordinates, see [paper-samplers.md](paper-samplers.md).

DPS and DAPS use complex tensors `(B, 1, 1, H, W)` at operator boundaries and two real channels internally. Both call the public conversion helpers, including supplied-noise and warm-start paths.

For DAPS, `tau` is measurement noise standard deviation per real coordinate: complex measurement noise has expected squared magnitude `2*tau**2`. The linear Gaussian approximation has precision `I/r**2 + A.H A/tau**2`. The Langevin likelihood gradient follows this convention without the previous extra factor of one half.

Finite Langevin remains the default inner kernel. Its step is bounded using the prior curvature and a supplied `operator_norm` upper bound on the linear operator spectral norm (default 1, appropriate to a unitary masked FFT). Supply a valid bound for other operators. This controls stiffness, but finite Langevin is still biased and does not certify posterior calibration. Sampling accepts an explicit `torch.Generator` and rejects nonfinite trajectories. PF integration finishes with terminal denoising.

`inner_kernel="gaussian"` enables perturb-and-solve draws using the adjoint and per-batch conjugate gradients, with convergence diagnostics and failure reporting. This is an exact draw to tolerance from the *local Gaussian approximation*, not from the learned posterior. The operator must be linear and batch independent, with an accurate adjoint. Gaussian inner draws are diagnostic: full draws from this broad approximation degraded the tested outer trajectories, despite accurate local solves. They are deliberately not the default.

A retrospective two-case screen with the same native EMA model and synthetic masked Fourier measurements found finite-Langevin DAPS and native-schedule DPS substantially better than the Gaussian-inner variant and historical unstable settings. These single draws overlap training stores and do not establish held-out performance or calibrated uncertainty. Keep named configurations separate; changes in starting sigma, measurement variance or noise temperature alter the experiment. An EDM denoiser can be used in a VP sampler with the explicit state and noise-level conversion documented above; passing VP states directly to an EDM denoiser without that conversion is incorrect.
