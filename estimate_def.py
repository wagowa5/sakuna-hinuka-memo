#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from typing import List, Tuple

R_097_103 = [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03]
R_99_105_OVER_102 = [99/102, 100/102, 101/102, 102/102, 103/102, 104/102, 105/102]


def s_def(def_eff: float, k: float = 27000.0) -> float:
    return 1.0 / (1.0 + def_eff / k)


def base_damage(atk: float, mult: float, def_eff: float, k: float = 27000.0) -> float:
    # B = mult * s(def_eff) * atk^2 / (atk + def_eff)
    return mult * s_def(def_eff, k) * (atk * atk) / (atk + def_eff)


def predicted_damages(atk: float, mult: float, def_eff: float, rs: List[float], k: float = 27000.0) -> List[int]:
    b = base_damage(atk, mult, def_eff, k)
    # floor をモデル化（浮動小数誤差対策で微小値を引く）
    return [int(math.floor(b * r + 1e-12)) for r in rs]


def loss_for_def(
    atk: float,
    mult: float,
    def_eff: int,
    observed: List[int],
    rs: List[float],
    k: float = 27000.0
) -> Tuple[int, List[int]]:
    preds = predicted_damages(atk, mult, float(def_eff), rs, k)
    # 観測が7個未満でも対応できるように、各観測値を「最も近い予測値」にマッチさせる
    total = 0
    for d in observed:
        total += min(abs(d - p) for p in preds)
    return total, preds


def estimate_def_seed(atk: float, mult: float, observed: List[int], k: float = 27000.0) -> int:
    """
    乱数中央値 r=1.00 とみなして、基礎ダメージBを観測の中央値から推定し、
    base_damage(atk,mult,def)=B を満たす def を二分探索で粗く求める。
    """
    obs_sorted = sorted(observed)
    # 7個揃っている想定なら中央値（4番目）を優先
    if len(obs_sorted) >= 5:
        target_b = float(obs_sorted[len(obs_sorted)//2])
    else:
        target_b = float(sum(obs_sorted)) / len(obs_sorted)

    def f(def_eff: float) -> float:
        return base_damage(atk, mult, def_eff, k)

    lo = 0.0
    hi = 1000.0
    # defが増えるほど f(def) は減るので、f(hi) <= target_b になるまでhiを拡張
    while f(hi) > target_b and hi < 1e7:
        hi *= 2.0

    # もし極端に小さいtargetで hi を超えても到達しない場合は上限で返す
    if f(hi) > target_b:
        return int(hi)

    # 二分探索
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if f(mid) > target_b:
            lo = mid
        else:
            hi = mid

    return int(round((lo + hi) / 2.0))


def estimate_def(
    atk: int,
    mult: float,
    observed: List[int],
    rs: List[float],
    k: float = 27000.0,
    window: int = 8000,
) -> Tuple[int, int, List[int]]:
    """
    1) 中央値近似で def の種を作る
    2) 種の周辺（±window）を総当たりして損失最小の def を採用
    """
    seed = estimate_def_seed(float(atk), mult, observed, k)
    start = max(0, seed - window)
    end = seed + window

    best_def = None
    best_loss = None
    best_preds: List[int] = []

    for d in range(start, end + 1):
        l, preds = loss_for_def(float(atk), mult, d, observed, rs, k)
        if (best_loss is None) or (l < best_loss) or (l == best_loss and d < best_def):
            best_loss = l
            best_def = d
            best_preds = preds
            if best_loss == 0:
                # 完全一致なら早期終了（ただし同損失のより小さいDEFがある可能性はあるので、
                # 仕様上はここで止めて良いならbreak。ここでは精密さ優先で続行しない場合のみbreakを外す）
                pass

    assert best_def is not None and best_loss is not None
    return best_def, best_loss, best_preds


def parse_int_list(s: str) -> List[int]:
    vals = []
    for part in s.split(","):
        p = part.strip()
        if p:
            vals.append(int(p))
    if not vals:
        raise ValueError("damages is empty")
    return vals


def main():
    ap = argparse.ArgumentParser(
        description="Estimate DEF_eff from ATK, multiplier, and observed damages with s(DEF)=1/(1+DEF/27000)."
    )
    ap.add_argument("--atk", type=int, required=True, help="ATK value (integer)")
    ap.add_argument("--mult", type=float, required=True, help="multiplier (e.g. 0.25 for normal, 1.6 for skill)")
    ap.add_argument("--damages", type=str, required=True, help="comma-separated observed damages (1..7 values)")
    ap.add_argument("--k", type=float, default=27000.0, help="K for s(DEF)=1/(1+DEF/K) (default: 27000)")
    ap.add_argument("--random", type=str, default="097_103",
                    choices=["097_103", "99_105_over_102"],
                    help="random multiplier set (default: 097_103)")
    ap.add_argument("--window", type=int, default=8000, help="search window around seed DEF (default: 8000)")
    ap.add_argument("--defdown", type=float, default=0.0,
                    help="DEF down rate (e.g. 0.165). If set, program also prints estimated original DEF.")
    args = ap.parse_args()

    observed = parse_int_list(args.damages)
    rs = R_097_103 if args.random == "097_103" else R_99_105_OVER_102

    best_def, best_loss, best_preds = estimate_def(
        atk=args.atk,
        mult=args.mult,
        observed=observed,
        rs=rs,
        k=args.k,
        window=args.window,
    )

    print("=== Result ===")
    print(f"ATK           : {args.atk}")
    print(f"mult          : {args.mult}")
    print(f"K             : {args.k}")
    print(f"random_set    : {args.random}  (r={rs})")
    print(f"observed      : {sorted(observed)}")
    print(f"best DEF_eff  : {best_def}")
    print(f"loss (L1)     : {best_loss}")
    print(f"predicted 7   : {best_preds}")

    if args.defdown and args.defdown > 0.0:
        # DEF_eff = DEF * (1 - defdown)
        orig = best_def / (1.0 - args.defdown)
        print(f"defdown       : {args.defdown}")
        print(f"estimated DEF : {orig:.3f}  (assuming DEF_eff=DEF*(1-defdown))")


if __name__ == "__main__":
    main()
