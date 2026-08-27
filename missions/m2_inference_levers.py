"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
CACHE_AVG_READS = 2.0
CACHE_WRITE_COST = 1.0


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    reasoning_cost = standard_cost = 0.0
    reasoning_wh = standard_wh = 0.0
    cache_enabled = pricing.cache_is_worth_it(CACHE_AVG_READS, CACHE_WRITE_COST)
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        is_reasoning = bool(int(num(r["is_reasoning"])))
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        opt_cost += pricing.request_cost(
            inp, out, pin, pout,
            cached_in=cached if cache_enabled else 0,
            batch=is_batch,
        )
        request_cost = pricing.request_cost(inp, out, pin, pout)
        request_wh = sustainability.wh_per_query(inp + out, is_reasoning=is_reasoning)
        if is_reasoning:
            reasoning_cost += request_cost
            reasoning_wh += request_wh
        else:
            standard_cost += request_cost
            standard_wh += request_wh

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"cache policy: {'enabled' if cache_enabled else 'disabled'} ({CACHE_AVG_READS:.0f} average reads, break-even > {CACHE_WRITE_COST / (1 - 0.10):.2f})")
        reasoning_share = reasoning_cost / (reasoning_cost + standard_cost) * 100 if reasoning_cost + standard_cost else 0.0
        traffic_share = sum(
            int(num(r["input_tokens"])) + int(num(r["output_tokens"]))
            for r in rows if bool(int(num(r["is_reasoning"])))
        ) / total_tokens * 100 if total_tokens else 0.0
        print(f"reasoning: {reasoning_share:.1f}% of cost, {traffic_share:.1f}% of tokens, {reasoning_wh:.1f} Wh vs {standard_wh:.1f} Wh standard")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_enabled": cache_enabled,
        "reasoning_cost": round(reasoning_cost, 4),
        "standard_cost": round(standard_cost, 4),
        "reasoning_wh": round(reasoning_wh, 2),
        "standard_wh": round(standard_wh, 2),
    }


if __name__ == "__main__":
    run()
