"""Generate the historical UK7T LUT without plotting or overwriting files.

Based on Olivier Mougin's notebook and Ian Driver's 2021/2022 adaptation,
recovered as datacheck/mprage_wand.py:create_LUT. Preserve original sequence
constants and five-shot steady state. Full grid is an explicit CPU operation
that historically takes about one hour; no generation runs on import.
"""

import argparse
from pathlib import Path

import numpy as np


def create_lut(max_b1: int = 200, max_t1: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    maxB1 = max_b1
    maxT1 = max_t1
    SSi = 3500
    TR = 6.3
    TFE = 224
    FAt1 = 5
    FAt2 = 2
    TI1_ = 725
    TIbis2 = 5
    TIbis = TI1_ + np.ceil(TFE / 2 * TR) + TIbis2
    TFE_bis = TFE / 2
    SSi10 = SSi * 10
    T = range(SSi10)
    TR10 = int(10 * TR)
    TR9 = TR10 - 1
    TIbis10 = int(TIbis * 10)
    TI1_10 = int(TI1_ * 10)
    TI4_10 = int(TIbis10 + TI1_10)
    TI0_10 = int(TI1_10 - np.ceil(TFE_bis * TR10))
    TI2_10 = int(TI1_10 + np.ceil(TFE_bis * TR10))
    TI3_10 = int(TI4_10 - np.ceil(TFE_bis * TR10))
    TI5_10 = int(TI4_10 + np.ceil(TFE_bis * TR10))
    if TI0_10 < 0:
        OSError("Error, wrong timing: please readjust sequence timing!")
    RF = np.zeros_like(T, dtype=bool)
    for i in range(TI0_10, TI2_10, 1):
        if not np.mod(i - TI0_10, TR10):
            RF[i] = True
    for i in range(TI3_10, TI5_10, 1):
        if not np.mod(i - TI3_10, TR10):
            RF[i] = True
    Mwm = 0.71
    FB = 180
    eff = 0.95
    cosFB = np.cos(np.deg2rad(eff * FB))
    B1m = range(1, maxB1 + 1, 1)
    T1m = range(1, maxT1 + 1, 1)
    LUT_INV1 = np.zeros((maxB1 + 1, maxT1 + 1))
    LUT_INV2 = np.zeros_like(LUT_INV1)
    LUT_MP2RAGE = np.zeros_like(LUT_INV1)
    LUT_PSIR = np.zeros_like(LUT_INV1)
    for T1tmp in T1m:
        expmT1 = np.exp(-0.1 / T1tmp)
        expTRT1 = np.exp(-(TR - 0.1) / T1tmp)
        for B1tmp in B1m:
            FA1 = FAt1 * B1tmp / 100
            cosFA1 = np.cos(np.deg2rad(FA1))
            sinFA1 = np.sin(np.deg2rad(FA1))
            FA2 = FAt2 * B1tmp / 100
            cosFA2 = np.cos(np.deg2rad(FA2))
            sinFA2 = np.sin(np.deg2rad(FA2))
            M = np.zeros((1, len(T)))
            for _shot in range(5):
                M[0, 1] = cosFB * M[0, 0]
                M[0, 120] = M[0, 1]
                M[0, TI0_10 - 1] = Mwm + (M[0, 120] - Mwm) * np.exp(-(float(TI0_10 - 1 - 120) / 10.0) / T1tmp)
                i = TI0_10
                while i <= TI1_10:
                    if RF[i]:
                        M[0, i] = Mwm + (cosFA1 * M[0, i - 1] - Mwm) * expmT1
                        i += 1
                    else:
                        M[0, i + TR9 - 1] = Mwm + (M[0, i - 1] - Mwm) * expTRT1
                        i += TR9
                INV1 = sinFA1 * M[0, TI1_10 - 1]
                i = TI1_10
                while i <= TI2_10:
                    if RF[i]:
                        M[0, i] = Mwm + (cosFA1 * M[0, i - 1] - Mwm) * expmT1
                        i += 1
                    else:
                        M[0, i + TR9 - 1] = Mwm + (M[0, i - 1] - Mwm) * expTRT1
                        i += TR9
                M[0, TI3_10 - 1] = Mwm + (M[0, TI2_10 - 1] - Mwm) * np.exp(-float(TI3_10 - 1 - TI2_10) / 10.0 / T1tmp)
                i = TI3_10
                while i <= TI4_10:
                    if RF[i]:
                        M[0, i] = Mwm + (cosFA2 * M[0, i - 1] - Mwm) * expmT1
                        i += 1
                    else:
                        M[0, i + TR9 - 1] = Mwm + (M[0, i - 1] - Mwm) * expTRT1
                        i += TR9
                INV2 = sinFA2 * M[0, TI4_10 - 1]
                i = TI4_10
                while i <= TI5_10:
                    if RF[i]:
                        M[0, i] = Mwm + (cosFA2 * M[0, i - 1] - Mwm) * expmT1
                        i += 1
                    else:
                        M[0, i + TR9 - 1] = Mwm + (M[0, i - 1] - Mwm) * expTRT1
                        i += TR9
                M[0, SSi10 - 1] = Mwm + (M[0, TI5_10 - 1] - Mwm) * np.exp(-(float(SSi10 - 1 - TI5_10) / 10.0) / T1tmp)
                M[0, 0] = M[0, SSi10 - 1]
            LUT_INV1[B1tmp, T1tmp] = INV1
            LUT_INV2[B1tmp, T1tmp] = INV2
    LUT_PSIR = LUT_INV1 / (abs(LUT_INV1) + abs(LUT_INV2))
    LUT_MP2RAGE = LUT_INV1 * LUT_INV2 / (LUT_INV1**2 + LUT_INV2**2)
    return (LUT_INV1, LUT_INV2, LUT_PSIR, LUT_MP2RAGE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-b1", type=int, default=200)
    parser.add_argument("--max-t1", type=int, default=3000)
    args = parser.parse_args()
    if args.max_b1 < 1 or args.max_t1 < 1:
        parser.error("grid limits must be positive")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        inv1, inv2, psir, mp2 = create_lut(args.max_b1, args.max_t1)
        np.savez(handle, inv1=inv1, inv2=inv2, psir=psir, mp2rage=mp2)


if __name__ == "__main__":
    main()
