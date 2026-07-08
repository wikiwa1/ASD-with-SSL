# Where does our JEPA ASD result sit vs. the literature?

Deep-research review (99 agents, 17 primary sources, 22/25 claims adversarially verified).
Question: how does a single-model frozen **audio I-JEPA + one-class Mahalanobis** reaching
**~0.82 mean AUC** (48 targets = 4 machines × 4 IDs × 3 SNRs, original MIMII split) compare
to published MIMII / DCASE 2020 Task 2 results?

## TL;DR
- **Clean win over the AE baseline.** 0.82 vs dense-AE **~0.72–0.74** under the *identical*
  original-MIMII protocol → **+8–10 AUC points**, a large gain for a label-free, frozen
  feature extractor. This is the only apples-to-apples comparison we have, and it's solid.
- **Not SOTA — but "SOTA" is a different protocol.** Published leaders use **DCASE 2020
  Task 2** (different eval IDs, noise mixing, 6 machine types, AUC + pAUC@0.1). Absolute
  AUCs are **not directly comparable**.
- **Even SOTA struggles on our hard machines** (fan/pump ~0.70–0.75), so 0.82 averaged
  across all SNRs incl. −6 dB is **competitive-but-not-leading**, not far off on hard types.
- **Every top system is ID-discriminative** (classification or flow density) — they exploit
  the machine-ID structure. Our I-JEPA uses no ID labels. **That's the gap and the lever.**
- **JEPA on industrial ASD is barely explored** — our approach is more *novel* than
  *record-beating*, which is arguably a better paper story.

## The protocol distinction (the crucial caveat)
| | Original MIMII (ours) | DCASE 2020 Task 2 (SOTA numbers) |
|---|---|---|
| Split | train normal-only / test held-out normal + all abnormal | dev/eval; **eval machine IDs differ from dev** |
| Machines | fan, pump, slider, valve (4) | + ToyCar, ToyConveyor (6) |
| Noise | explicit −6 / 0 / 6 dB SNR | environmental-noise mixing, 16 kHz mono, no SNR split |
| Metric | AUC | **AUC and pAUC@0.1** |

→ A "0.82 over 48 targets" cannot be set equal to a DCASE AUC. Report both, and ideally
**also evaluate under the DCASE protocol with pAUC** to place it on the real leaderboard.

## Reference numbers (verified)
**MIMII dense-AE baseline (Purohit et al. 2019, same protocol as ours), per-machine AUC at
6 / 0 / −6 dB:** fan 0.94/0.84/0.70 · pump 0.81/0.74/0.68 · slide rail 0.90/0.80/0.70 ·
valve 0.67/0.61/0.53. Mean ≈ **0.73–0.74**. Valve is hardest everywhere.

**DCASE 2020 Task 2 SOTA — Giri et al. (Amazon, winner): self-supervised machine-ID
classification + Group MADE density estimation.** Per-machine AUC/pAUC (ensemble, dev set):
slider 94.0/83.0 · valve 93.4/76.2 · **fan 70.3/53.2 · pump 75.0/65.9** (+ ToyCar 80.2,
ToyConveyor 75.1). Single-model variant only marginally lower. The official AE baseline team
ranked **33rd of 40**; top systems ~0.90+ mean AUC vs baseline ~0.65 (DCASE dev).

Other strong directions: **Daniluk** (Samsung, AE+WaveNet ensemble, rank 3); **Primus**
(CP-JKU, outlier-exposed classifiers, rank 5); **Dohi et al. ICASSP'21** flow-based
self-supervised density estimation (+4.6–5.8% AUC over flow baselines); recent **large
pretrained SSL** (wav2vec2 / HuBERT / WavLM / BEATs) as 2023–2025 strong baselines.

## What to try next (highest leverage first)
1. **Add an ID-discriminative head** (machine-ID / section classification on the JEPA
   encoder) — the single common thread among all winners.
2. **Normalizing-flow likelihood scoring** instead of Mahalanobis (Dohi et al.).
3. **Benchmark vs large pretrained SSL** (BEATs / wav2vec2) — current strong baselines.
4. **Evaluate under the DCASE 2020 Task 2 protocol** (report pAUC@0.1) for a leaderboard
   comparison.

## Caveats / what the verifier rejected
- **Protocol mismatch dominates** — don't equate the 0.82 with DCASE AUCs.
- A claimed Giri "mean AUC ≈ 79.7" headline was **refuted** — cite the per-machine table.
- A 2023 contrastive paper's "beats SOTA" claim was **refuted** — the SSL/contrastive frontier
  is not cleanly settled.
- Confirm we averaged the **4 public MIMII IDs** (id_00/02/04/06) — the standard subset.

## Sources (primary, verified)
- MIMII baseline: arXiv:1909.09347 · github.com/MIMII-hitachi/mimii_baseline
- DCASE 2020 Task 2: Koizumi et al. arXiv:2006.05822 · dcase.community results page
- Giri et al. (winner): DCASE2020Workshop_Giri_66.pdf
- Flow SSL: Dohi et al. arXiv:2103.08801
- Contrastive ID: Guan et al. arXiv:2304.03588 · Large-SSL: Han et al. arXiv:2508.12230
- A-JEPA (audio JEPA): arXiv:2311.15830

_Full machine-readable report + verbatim quotes:_
`/tmp/claude-105125/-pscratch-sd-d-dfarough-ASD-with-SSL/74eb0515-dbe9-48ec-8612-21031bd51e1f/tasks/wp8qd3ppg.output`
