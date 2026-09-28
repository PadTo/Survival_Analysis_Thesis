# Two modelling bugs (plain language, for thesis text)

Use this note, not the code comments. Numbers are for the 14-day personal
frame after 56-day user-span filtering. Code: `src/data_processing.py`
(entry filter) and `Coding/R/survival_testing.R` (`cox_hazard`).

---

## Bug 1 — counting started too early (immortal time)

**What we wanted.** Keep only users with ≥ 56 days of activity. Then only
score them from the first time an event is even *allowed*.

**What we did wrong.** We dropped short-span *users*, but still put every
kept user into the model from their own day 0.

Labels are lagged one interval: a row that starts at day `t` predicts
churn in the *next* 14 days, `[t+14, t+28)`. So:

| row starts | asks “churn in …?” | allowed under a 56-day span rule? |
|---|---|---|
| 0 | [14, 28) | no |
| 14 | [28, 42) | no |
| 28 | [42, 56) | no |
| 42 | [56, 70) | **yes — first modelled row** |
| 56 | [70, 84) | yes |

Rows 0 / 14 / 28 can never be events for these users. They are free
non-events. They also sit at the start of each user’s table, so 28–56-day
lookbacks and “drift” were empty and filled with 0. That is why drift
equaled the current 28-day value on exactly 6,000 × 2 rows: lag of the
*table*, not missing calendar data.

There is no activity before each user’s day 0. Days 0–42 *are* real
history. We still compute features on those rows internally so the day-42
row’s lookbacks are filled. We just **do not put those rows in the model**.

**Fix.** After all lags/shifts, drop exported rows with
`interval_start < 56 − 14 = 42` (28-day grid: start at 28). Events stayed
2,774. About 18,000 empty-risk rows went away. AUC fell because those
rows were easy, guaranteed negatives.

Entry is **42**, not 56. Day 42 is the first row whose *question* is
“event after day 56?”.

---

## Bug 2 — Cox interval hazard of 0 on that first row (not splines)

Cox (M0 / M1) turns a fit into a 14-day probability using the **jump in
cumulative baseline hazard** on that window:

`ΔΛ = Λ(end) − Λ(start)`, then `h = 1 − exp(−ΔΛ × exp(lp))`.

Breslow only jumps at event times. After bug 1, the first event time in
the table is **56**. The first modelled window is **42 → 56**.

The interpolator had no Λ(42). It copied the first table value, which is
already Λ(56). Then

`Λ(56) − Λ(42) = Λ(56) − Λ(56) = 0`

so every Cox prediction on the busiest interval (184 events / 6,000 rows)
was exactly 0. Cloglog (M2a / M2b) was fine: it has a free intercept per
interval. Splines were not the issue.

**Fix.** Before the first event time in the Breslow table, set `Λ(t) = 0`
(nothing has jumped yet). Then the first window is `Λ(56) − 0`, the next
is `Λ(70) − Λ(56)`, then `Λ(84) − Λ(70)`, and so on.

Do **not** draw a straight line from (time 0, Λ = 0) to the first jump:
that under-states the first interval and wrecked calibration.

**After the fix.** Mean predicted hazard on the entry interval ≈ observed
rate (~0.03). M0/M1 AUC moved back in line with M2a/M2b (~0.68). XGBoost
never used this baseline function.

---

## What to cite as current results

Live CSVs in `Coding/Data/final/` after both fixes. Headline OOF means:
M0 0.680, M1 0.682, M2a 0.680, M2b 0.683, XGB 0.673 (AUC); Brier ~0.0215;
calibration slopes ~0.89–1.00. Full file list: `Coding/Data/final/RESULTS_MANIFEST.md`.
