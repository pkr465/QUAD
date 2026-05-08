---
name: quad-profile
description: Use when the user asks to "profile a model", "find bottlenecks", "measure NPU latency", "run linting" or "QHAS profile", or invokes /quad-profile. Picks the right profiling level + renders a rich latency/utilisation summary.
trigger: "profile"
---

# QUAD Profiling Skill

Run platform profilers on a model and surface the results in a way
that makes optimisation opportunities obvious.

## Profiling levels (pick one based on the user's question)

| Question                                | Level     |
| :---                                    | :---      |
| "How fast is it on NPU?"                | detailed  |
| "What's the p99 latency?"               | detailed  |
| "Where is it slow?" / "find bottlenecks"| linting   |
| "Why aren't I getting peak NPU usage?"  | linting   |
| "I need a Chrome trace"                 | qhas      |
| "Just give me a rough number"           | basic     |

## Steps

1. Call `mcp__quad__profile_workload` with the right `profiling_level`,
   `runtime` (default `auto`), and `duration_s` (default 10, longer
   for thermally-throttling sustained tests).

2. Format with `quad.ui.format_profile(report)` — gives a latency
   table (mean / p50 / p95 / p99 / min / max), throughput row, and
   utilisation bars per compute unit.

3. **Linting mode specifically:**
   - Highlight the `linting_total_cycles` and `linting_bottleneck_count`
   - Show top 3 ops by cycle count with their `overlap_ratio` and any
     `optimization_hint`
   - Surface the `linting_optimization_hints` list — these are
     actionable rewrites the user can apply

4. **QHAS mode specifically:**
   - Mention `qhas_chrometrace_path` — the user can drag this into
     `chrome://tracing` for a kernel-level visualisation
   - Note that QHAS requires the `snpe-dlc-graph-prepare` tool +
     `qnn-profile-viewer` (run `quad doctor --real-mode` to confirm
     they're available)

5. Auto-suggest follow-ups using `quad.suggestions.suggest_optimisations`
   for any detected bottlenecks.

6. End with 1-2 tips from `quad.tips.get_tips_for("profile")`.

## Output style

For detailed profile:
```
### Profile — runtime: NPU · level: detailed

**Latency**
| Statistic | Value (ms) |
| :---      | ---:       |
| Mean      | 2.56       |
| p50       | 2.20       |
| p95       | 4.68       |
| p99       | 6.28       |

**Throughput:** 388 FPS · **Power:** 2000 mW · **Memory:** peak 50 MB

**Compute utilisation**
`███████████████████░` 95% NPU
`░░░░░░░░░░░░░░░░░░░░`  0% GPU
`█░░░░░░░░░░░░░░░░░░░`  5% CPU
```

For linting:
```
### Profile — runtime: NPU · level: linting

Total cycles: 3,753,939
Bottlenecks: 1 op with low HTP parallelism

⚠ **1 bottleneck op detected**

| # | Op       | Cycles    | Overlap | Hint                                |
| -: | :--      | --:       | --:     | :---                                |
| 4  | sub_op   | 2,165,162 | 21%     | Replace Sub with Conv on HTP v68+   |
```

## When the user provides a model that hasn't been converted yet

- Run `quad-convert` first (or remind the user to).
- Profiling against an .onnx/.pt directly works in mock mode but real
  mode needs the converted .dlc / .bin.
