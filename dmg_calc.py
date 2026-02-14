#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from typing import List

R_097_103 = [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]
R_99_105_OVER_102 = [99/102, 100/102, 101/102, 102/102, 103/102, 104/102, 105/102]


def s_def(def_eff: float, k: float) -> float:
    # s(DEF_eff)=1/(1+DEF_eff/K)
    return 1.0 / (1.0 + def_eff / k)


def base_damage(atk: float, mult: float, def_eff: float, k: float) -> float:
    # B = mult * s(def_eff) * atk^2 / (atk + def_eff)
    return mult * s_def(def_eff, k) * (atk * atk) / (atk + def_eff)


def calc_damages(atk: int, mult: float, def_eff: float, k: float, rs: List[float]) -> List[int]:
    b = base_damage(float(atk), mult, def_eff, k)
    # floorモデル（浮動小数誤差ケア）
    return [int(math.floor(b * r + 1e-12)) for r in rs]


def main():
    ap = argparse.ArgumentParser(description="Quick damage calculator (7 RNG outcomes).")
    ap.add_argument("--atk", type=int, required=True, help="ATK")
    ap.add_argument("--mult", type=float, required=True, help="multiplier (e.g. 0.25 normal, 1.6 skill)")
    ap.add_argument("--def_enemy", type=float, help="enemy DEF (original DEF before defdown). Provide either --def or --def_eff.")
    ap.add_argument("--def_eff", type=float, help="enemy effective DEF (after defdown). Provide either --def or --def_eff.")
    ap.add_argument("--defdown", type=float, default=0.0, help="DEF down rate (e.g. 0.165). Used only if --def is given.")
    ap.add_argument("--k", type=float, default=27000.0, help="K in s(DEF_eff)=1/(1+DEF_eff/K). default 27000")
    ap.add_argument("--random", choices=["097_103", "99_105_over_102"], default="097_103",
                    help="RNG set. default 097_103 (=0.97..1.03)")
    args = ap.parse_args()

    if args.def_eff is None and args.def_enemy is None:
        raise SystemExit("ERROR: provide --def_enemy or --def_eff")
    if args.def_eff is not None and args.def_enemy is not None:
        raise SystemExit("ERROR: provide only one of --def_enemy or --def_eff")

    rs = R_097_103 if args.random == "097_103" else R_99_105_OVER_102

    if args.def_eff is not None:
        def_eff = float(args.def_eff)
        def_orig = None
    else:
        def_orig = float(args.def_enemy)
        def_eff = def_orig * (1.0 - float(args.defdown))

    b = base_damage(float(args.atk), float(args.mult), def_eff, float(args.k))
    outs = calc_damages(args.atk, args.mult, def_eff, args.k, rs)

    print("=== Damage Calc ===")
    print(f"ATK        : {args.atk}")
    print(f"mult       : {args.mult}")
    print(f"K          : {args.k}")
    print(f"random_set : {args.random}  (r={rs})")
    if def_orig is not None:
        print(f"DEF(orig)  : {def_orig}")
        print(f"defdown    : {args.defdown}")
    print(f"DEF_eff    : {def_eff}")
    print(f"base(B)    : {b:.6f}")
    print(f"damages(7) : {outs}")
    print(f"min..max   : {min(outs)} .. {max(outs)}")


if __name__ == "__main__":
    main()
