# Calibration of ttm_c256_h96_ft_2026-09-23 on our own arena games (2026-09-25)

The ladder calibration (`../calibration.json`) did not help: in the rematch against feed15 with
the 09-13 model it went 19W-141L (−$265 a game) over the 160 games played before it was stopped,
against 24W-174L uncalibrated. This fit uses the games our agent actually plays instead:
- **pool:** feed15 (09-13 model) against plain Mohui (200 games) and against Hazel, Copper,
  Orchard, Willow and Mohui13 (40 each). This is the feedfix batch, scored from feed15's side.
- **mirror:** feed15 with the 09-23 model against feed15 with the 09-13 model (the model0923
  batch, 200 games), scored from both sides.

Each game was rebuilt from its arena trace by replaying the recorded actions through the engine;
every replay reproduced the recorded final cash. Both models were scored on every turn from 256
to 714, 367,200 origins in total. The run used one Colab T4 High-RAM VM (`calibown`, 47 min); a
second T4 was refused. The fit has guards: a threshold counts only if the 09-13 model fires on
at least 100 origins and the product is sold at at least 0.5 % of origins, and factors are
clipped to 0.5–2.

| file | what |
|---|---|
| `calibration.json` | pool + mirror (used for the model0923own rematch) |
| `calibration_pool.json`, `calibration_mirror.json` | the two subsets on their own |
| `fit_report*.txt` | firing rate, precision and AUC per product and threshold |

**Findings.**
- **Ranking.** On our games the 09-23 model ranks the most-traded products worse than the 09-13
  model. For `score_4` (the front-run trigger), wheat drops 0.668 → 0.635 (pool) and 0.697 →
  0.648 (mirror), strawberry 0.795 → 0.773 and fertilizer 0.782 → 0.771. It is better only on
  carrot and melon, which are rarely sold here. On the ladder games it was better or equal
  almost everywhere. So it predicts top ladder players better and Mohui-family agents worse, and
  AUC cannot be changed by any calibration.
- **What our opponents sell.** They never sell tomatoes or eggs (base rate 0 %), so those
  factors stay 1.
- **Factors.** They differ a lot from the ladder fit. For example, wheat is 0.65 on `score_24`
  and 1.43 on `units_24`, strawberry `units_24` is 1.27, fertilizer about 1.2, and carrot and
  melon `score_4` are 1.86 and 1.71. The pool and mirror subsets agree roughly.
