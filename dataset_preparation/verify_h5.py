"""Compare HDF5 arrays without assuming container-byte identity or loading volumes."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _attrs_equal(left: h5py.AttributeManager, right: h5py.AttributeManager) -> bool:
    return set(left) == set(right) and all(np.array_equal(left[name], right[name]) for name in left)


def compare_h5(
    reference: str | Path,
    candidate: str | Path,
    subjects: list[str] | None = None,
    sample_slices: int | None = None,
    check_attributes: bool = False,
    check_layout: bool = False,
    rtol: float = 0.0,
    atol: float = 0.0,
    dataset_paths: list[str] | None = None,
) -> dict:
    """Compare structure, shape, dtype and every value unless sampling is explicit.

    Subjects are exact root keys, or subject keys in combined/coilwise groups.
    Attribute and physical layout checks are independent opt-ins. Numeric
    equality is exact by default; corresponding NaNs compare equal. Explicit
    tolerances affect equivalence, while bit_exact still reports exact equality.
    """
    if sample_slices is not None and sample_slices <= 0:
        raise ValueError("sample_slices must be positive")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError("Tolerances must be finite and nonnegative")
    issues: list[dict[str, object]] = []
    checked_datasets = 0
    checked_slices = 0
    checked_values = 0
    values_exact = True
    maximum_difference = 0.0
    dataset_reports = []
    with h5py.File(reference, "r") as left, h5py.File(candidate, "r") as right:

        def inventory(handle: h5py.File) -> dict:
            result = {"/": handle}
            handle.visititems(lambda name, obj: result.__setitem__(name, obj))
            if subjects is not None:
                wanted = set(subjects)
                result = {
                    name: obj
                    for name, obj in result.items()
                    if name == "/"
                    or name in ("coilwise", "combined")
                    or name.split("/")[0] in wanted
                    or (name.split("/")[0] in ("coilwise", "combined") and len(name.split("/")) > 1 and name.split("/")[1] in wanted)
                }
            if dataset_paths is not None:
                wanted = {path.strip("/") for path in dataset_paths}
                result = {
                    name: obj
                    for name, obj in result.items()
                    if name == "/" or any(name == path or path.startswith(name + "/") for path in wanted)
                }
            return result

        first, second = inventory(left), inventory(right)
        if set(first) != set(second):
            issues.append({"structure": {"missing": sorted(set(first) - set(second)), "extra": sorted(set(second) - set(first))}})
        if subjects is not None:
            for subject in subjects:
                if not any(subject in name.split("/") for name in first):
                    issues.append({"missing_requested_subject": subject})
        if dataset_paths is not None:
            for path in dataset_paths:
                if path.strip("/") not in first:
                    issues.append({"missing_requested_dataset": path})
        for name in sorted(set(first) & set(second)):
            a, b = first[name], second[name]
            if isinstance(a, h5py.Dataset) != isinstance(b, h5py.Dataset):
                issues.append({"path": name, "reason": "object type"})
                continue
            if check_attributes and not _attrs_equal(a.attrs, b.attrs):
                issues.append({"path": name, "reason": "attributes"})
            if not isinstance(a, h5py.Dataset):
                continue
            checked_datasets += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                issues.append({"path": name, "reason": "shape or dtype", "reference_shape": a.shape, "candidate_shape": b.shape})
                continue
            if check_layout and any(
                getattr(a, key) != getattr(b, key)
                for key in ("chunks", "compression", "compression_opts", "shuffle", "fletcher32", "scaleoffset")
            ):
                issues.append({"path": name, "reason": "container layout"})
            indices: list[int | tuple[()]] = list(range(a.shape[0])) if a.ndim else [()]
            if a.ndim and sample_slices is not None and len(indices) > sample_slices:
                indices = np.unique(np.linspace(0, len(indices) - 1, sample_slices).astype(int)).tolist()
            mismatched_slices = 0
            different_values = 0
            local_maximum = 0.0
            exact = True
            for index in indices:
                checked_slices += 1
                x, y = a[index], b[index]
                checked_values += np.size(x)
                numeric = np.issubdtype(a.dtype, np.inexact)
                same = np.array_equal(x, y, equal_nan=numeric)
                byte_equal = same if a.dtype.hasobject else np.asarray(x).tobytes() == np.asarray(y).tobytes()
                exact = exact and byte_equal
                if not same:
                    different_values += (
                        int(np.count_nonzero(~np.isclose(x, y, rtol=0, atol=0, equal_nan=True)))
                        if numeric
                        else int(np.count_nonzero(x != y))
                    )
                    if numeric or np.issubdtype(a.dtype, np.integer):
                        calculation_dtype = np.complex128 if np.issubdtype(a.dtype, np.complexfloating) else np.float64
                        with np.errstate(invalid="ignore"):
                            differences = np.abs(np.asarray(x, dtype=calculation_dtype) - np.asarray(y, dtype=calculation_dtype))
                        finite = differences[np.isfinite(differences)]
                        if finite.size:
                            local_maximum = max(local_maximum, float(finite.max()))
                        if np.isinf(differences).any() or np.any(np.isnan(x) != np.isnan(y)):
                            local_maximum = float("inf")
                equivalent = same or (numeric and np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True))
                if not equivalent:
                    mismatched_slices += 1
            if mismatched_slices:
                issues.append({"path": name, "reason": "values", "mismatched_slices": mismatched_slices})
            values_exact = values_exact and exact
            maximum_difference = max(maximum_difference, local_maximum)
            dataset_reports.append(
                {
                    "path": name,
                    "bit_exact": exact,
                    "different_values": different_values,
                    "checked_slices": len(indices),
                    "max_abs_difference": local_maximum,
                }
            )
    return {
        "equal": not issues,
        "bit_exact": values_exact and not issues,
        "rtol": rtol,
        "atol": atol,
        "max_abs_difference": maximum_difference,
        "mode": "sampled" if sample_slices is not None else "full",
        "subject_subset": subjects,
        "dataset_path_subset": dataset_paths,
        "attributes_checked": check_attributes,
        "layout_checked": check_layout,
        "checked_datasets": checked_datasets,
        "checked_slices": checked_slices,
        "checked_values": int(checked_values),
        "datasets": dataset_reports,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--sample-slices", type=int, help="Explicitly sample instead of checking every slice")
    parser.add_argument("--check-attributes", action="store_true")
    parser.add_argument("--check-layout", action="store_true")
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--dataset-paths", nargs="+", help="Exact HDF5 dataset paths; reference may contain additional subjects")
    parser.add_argument("--report", type=Path, help="Optional new JSON report path; existing paths are refused")
    args = parser.parse_args()
    report = compare_h5(
        args.reference,
        args.candidate,
        args.subjects,
        args.sample_slices,
        args.check_attributes,
        args.check_layout,
        args.rtol,
        args.atol,
        args.dataset_paths,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["equal"] else 1)


if __name__ == "__main__":
    main()
