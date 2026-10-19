"""Execute frozen reference transitions with an EDM oracle and compare traces."""

import ast
import hashlib
import json
import math
import types
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jsonargparse import ArgumentParser

from score_qmri.sampling.ancestral_dps import AncestralDPSConfig, AncestralDPSSampler
from score_qmri.sampling.daps_sampler import to_2ch, to_complex
from score_qmri.sampling.edm_sampler import karras_sigmas
from score_qmri.sampling.paper_daps import PaperDAPSConfig, PaperDAPSSampler

parser = ArgumentParser(description=__doc__)
parser.add_argument("--references", type=Path, required=True, help="Directory containing the two pinned reference source trees")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
ROOT = args.references
manifest = json.loads(Path(__file__).with_name("reference_sampler_sources.json").read_text())
for relative, expected in manifest.items():
    actual = hashlib.sha256((args.references / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Reference source hash mismatch: {relative}")
torch.set_num_threads(2)


def declarations(path: Path, namespace: dict[str, Any]) -> dict[str, Any]:
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.Assign))]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class Oracle(torch.nn.Module):
    sigma_min = 0.001
    sigma_max = 20.0
    device = torch.device("cpu")

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def denoise(self, x: torch.Tensor, sigma: torch.Tensor | float, kind: torch.Tensor | None = None) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        return x * 0.04 / (0.04 + sigma.reshape(-1, 1, 1, 1) ** 2)

    def score(self, x: torch.Tensor, sigma: torch.Tensor | float, kind: torch.Tensor | None = None) -> torch.Tensor:
        sigma = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
        return -x / (0.04 + sigma.reshape(-1, 1, 1, 1) ** 2)

    def clamp_x0(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("No clipping in parity controls")


class ComplexIdentity:
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x,)

    @property
    def H(self) -> "ComplexIdentity":
        return self


def dps() -> dict[str, Any]:
    base = ROOT / "diffusion-posterior-sampling/guided_diffusion"
    processors = declarations(base / "posterior_mean_variance.py", {"np": np, "torch": torch, "ABC": ABC, "abstractmethod": abstractmethod})
    reference = declarations(
        base / "gaussian_diffusion.py",
        {
            "np": np,
            "torch": torch,
            "math": math,
            "get_mean_processor": processors["get_mean_processor"],
            "get_var_processor": processors["get_var_processor"],
        },
    )
    condition = declarations(base / "condition_methods.py", {"torch": torch, "ABC": ABC, "abstractmethod": abstractmethod})

    class Operator:
        def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return x

    conditioner = condition["PosteriorSampling"](Operator(), types.SimpleNamespace(__name__="gaussian"), scale=0.03)
    cfg = AncestralDPSConfig(num_steps=16, guidance_scale=0.03)
    oracle = Oracle()
    model = AncestralDPSSampler(oracle, cfg)
    sigmas = karras_sigmas(16, sigma_min=oracle.sigma_min, sigma_max=20.0, rho=7.0, device=oracle.device).double()
    ascending = sigmas.flip(0).numpy()
    alpha_bar = 1 / (1 + ascending**2)
    betas = 1 - alpha_bar / np.r_[1.0, alpha_bar[:-1]]
    out = {}
    for head in ["start_x", "epsilon"]:
        ref = reference["DDPM"](
            use_timesteps=set(range(16)),
            betas=betas,
            model_mean_type=head,
            model_var_type="fixed_small",
            dynamic_threshold=False,
            clip_denoised=False,
            rescale_timesteps=False,
        )

        def edm_adapter(z: torch.Tensor, t: torch.Tensor, head: str = head) -> torch.Tensor:
            a = torch.as_tensor(alpha_bar, dtype=z.dtype)[t].reshape(-1, 1, 1, 1)
            sigma = torch.as_tensor(ascending, dtype=z.dtype)[t]
            clean = oracle.denoise(z / a.sqrt(), sigma)
            return clean if head == "start_x" else (z - a.sqrt() * clean) / (1 - a).sqrt()

        initial = torch.randn(1, 2, 2, 2, generator=torch.Generator().manual_seed(79), dtype=torch.float64)
        y = initial * 0.1
        state = initial.clone()
        trace = []
        torch.manual_seed(81)
        for idx in range(15, -1, -1):
            state = state.detach().requires_grad_(True)
            proposal = ref.p_sample(edm_adapter, state, torch.tensor([idx]))
            state, _ = conditioner.conditioning(state, proposal["sample"], proposal["pred_xstart"], y)
            state = state.detach()
            trace.append(state.clone())
        local_state = initial.clone()
        local_trace = []
        rng = torch.Generator().manual_seed(81)
        schedule = torch.cat([sigmas, sigmas.new_zeros(1)])
        for s, t in zip(schedule[:-1], schedule[1:], strict=False):
            eps = torch.randn(initial.shape, dtype=initial.dtype, generator=rng)
            local_state = model._step(local_state, torch.tensor([1]), ComplexIdentity(), to_complex(y), float(s), float(t), noise=eps)
            local_trace.append(local_state.clone())
        out[head] = {
            "max_trace_abs_error": max(float((a - b).abs().max()) for a, b in zip(trace, local_trace, strict=False)),
            "final_abs_error": float((state - local_state).abs().max()),
            "steps": 16,
            "variance": "fixed_small",
            "clip_denoised": False,
            "guidance": "official norm conditioning in VP coordinates",
        }
    return out


def daps() -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(ROOT / "DAPS"))
    from cores.mcmc import MCMCSampler
    from cores.scheduler import DiffusionPFODE, get_diffusion_scheduler
    from cores.trajectory import Trajectory

    reference = declarations(
        ROOT / "DAPS/sampler.py",
        {
            "torch": torch,
            "nn": torch.nn,
            "get_diffusion_scheduler": get_diffusion_scheduler,
            "DiffusionPFODE": DiffusionPFODE,
            "MCMCSampler": MCMCSampler,
            "Trajectory": Trajectory,
        },
    )
    operator_path = ROOT / "DAPS/forward_operator/__init__.py"
    operator_tree = ast.parse(operator_path.read_text())
    operator_tree.body = [node for node in operator_tree.body if isinstance(node, ast.ClassDef) and node.name == "Operator"]
    operators = {"torch": torch, "ABC": ABC, "abstractmethod": abstractmethod}
    exec(compile(operator_tree, str(operator_path), "exec"), operators)

    class Identity(operators["Operator"]):
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return x

    oracle = Oracle()
    cfg = PaperDAPSConfig(tau=0.2, anneal_steps=8, reverse_steps=5, langevin_steps=7)
    initial = torch.randn(1, 2, 2, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(47))
    y = initial * 0.1
    out = {}
    for prior, refprior in [("gaussian", "gaussian"), ("score", "exact")]:
        ref = reference["DAPS"](
            {"name": "edm", "num_steps": 8, "sigma_max": 20.0, "sigma_min": 0.1, "timestep": "poly-7"},
            {"name": "edm", "num_steps": 5, "sigma_min": 0.01, "timestep": "poly-7"},
            {"num_steps": 7, "lr": cfg.langevin_lr, "tau": math.sqrt(2) * cfg.tau, "prior_solver": refprior, "prior_sigma_min": 0.001},
        )
        torch.manual_seed(39)
        expected = ref.sample(oracle, 20 * initial, Identity(), y, record=True)
        from dataclasses import replace

        local = PaperDAPSSampler(oracle, replace(cfg, prior=prior))
        actual = to_2ch(
            local.sample(ComplexIdentity(), to_complex(y), 1, noise=to_complex(initial), generator=torch.Generator().manual_seed(39))
        )
        out[prior] = {
            "final_abs_error": float((actual - expected).abs().max()),
            "sigma_grid_abs_error": float((local.last_sigmas - ref.annealing_scheduler.sigma_steps).abs().max()),
            "steps": 8,
            "inner_steps": 7,
            "reverse_steps": 5,
            "reference_tau_adapter": "sqrt(2)*tau compensates the reference squared-loss factor two; no algorithm edits",
        }
    return out


result = {"official_dps": dps(), "official_daps": daps()}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as handle:
    json.dump(result, handle, indent=2)
print(json.dumps(result, indent=2))
