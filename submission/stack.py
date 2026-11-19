"""SimulacraBench submission: a per-item stacked ensemble over the GIVEN block.

The delivered frame hands you every visible respondent complete, so each
instrument is a fully observed multi-task classification problem: one row per
respondent, the GIVEN block as features, and one categorical target per PREDICT
item -- with the gate sentinel as an ordinary category, because being skipped is
a real answer the grader scores.

Five families of estimator are fitted on the visible rows:

  marginal    the crowd, per item. The thing to beat.
  backoff     Dirichlet backoff down the 1..3 GIVEN columns carrying the most
              mutual information with the item -- the starter's estimator.
  logistic    one multinomial logistic regression per item, all of them fitted
              as a single batched GPU problem sharing one design matrix, at
              several L2 strengths at once.
  trunk       the same, through a narrow shared trunk. A low-rank respondent
              code is what a latent-trait / IRT model is, and it lets an item
              borrow strength from every other item.
  latent      a latent class per respondent: P(class | GIVEN) is a softmax and
              P(answer | class) a free table, so the conditional is a mixture
              rather than a single softmax -- the shape a latent trait induces.
  knn         the k visible respondents who match this one on the most GIVEN
              columns, voting. The one local, non-parametric member.

and each of them again as a `+gate` variant, corrected by the skip logic:
P(sentinel) comes from the gate -- an indicator when the parent is GIVEN, the
parent's own predicted distribution when it is not -- and the answer
distribution is renormalised under it.

`+gateC` is that correction again with the gate's *fidelity* read off the GIVEN
block rather than pooled into two numbers. UNHCR needs it: five multi-select
families code a skip differently in each round, so P(sentinel | not asked) is 0
on one wave and 1 on the other, and no single number is right for either. The
two readings sit on the roster together and the data picks, per item.

Last, one exponent per item on the pooled distribution, fitted on held-out
cells. A linear pool sits between its members, so it is flatter than any of
them; how much that costs is measured rather than assumed, and 1.0 is in the
grid.

None of them is trusted in advance. Every family is fitted K-fold, and the
out-of-fold likelihood of the answer that was actually there decides, per item,
the mixture weight each family gets (EM on the simplex). An estimator that is
wrong about an instrument -- the gate correction on an instrument whose skip
logic the data only approximately obeys, say -- is weighted out by the data
rather than by a guess made here. The fold models also read out the rows being
predicted, so the models that earned the weights are the models the weights are
applied to, and averaging over the folds costs nothing and removes variance.

Everything is fitted at call time on the frame, which is where the only real
answers are; nothing is bundled and nothing is downloaded. The wall-clock
budget covers module import and all three instruments together, so each call
takes a share of what is left and every stage degrades to the stage before it:
a family that runs out of time is filled with the crowd marginal rather than
dropped, which keeps the roster -- and therefore the weights -- aligned.
"""

import json
import os
import re
import time
import zlib

import numpy as np

_IMPORTED_AT = time.time()


def _process_started_at():
    """When this container's python started, not when this module imported.

    The grader's clock is wall time for the whole run, and everything before
    the first line of this file -- container start, the interpreter, numpy,
    pandas and torch coming in, the orchestrator materialising the first frame
    -- is spent out of the same 900 s. v8 measured the budget from
    `_IMPORTED_AT`, which silently handed that prologue back as free time and
    then paid for it again in a 0.85 safety factor. Reading the process's own
    start time turns the guess into a measurement: the budget is counted from
    where the grader counts it, and the factor only has to cover overshoot.

    /proc is Linux and the runtime is Linux; anywhere it is not, this falls
    back to the import time, which is the v8 behaviour.
    """
    try:
        with open("/proc/self/stat") as handle:
            fields = handle.read().rsplit(") ", 1)[1].split()
        ticks = float(fields[19])                       # starttime, field 22
        with open("/proc/uptime") as handle:
            uptime = float(handle.read().split()[0])
        hz = float(os.sysconf("SC_CLK_TCK"))
        age = uptime - ticks / hz
        if 0.0 <= age < 3600.0:
            return _IMPORTED_AT - age
    except Exception:
        pass
    return _IMPORTED_AT


_STARTED_AT = _process_started_at()
_PROLOGUE = _IMPORTED_AT - _STARTED_AT
_CALLS = 0
_SEEN = set()

# Tunables. SB_CONFIG={"json": ...} overrides any of them, for local sweeps.
CONFIG = {
    # v9. Four, because the clock now reaches it. A fold is worth about
    # +0.0013 on a ceiling-bearing fixture and it is not only a fold: at four
    # each fold model is fitted on 75% of the visible rows instead of 66%, and
    # the bag that ships averages five builds instead of four. The loop is
    # time-driven and stops early if the room is not there, so this is a
    # ceiling and not a promise -- what it costs when the clock is short is
    # that EM sees three quarters of the rows instead of all of them.
    # v10. Blocks per pass, and this is the *coverage* unit rather than a
    # clock knob. v9 raised it to 4 for the extra training rows a fold model
    # gets and shipped a configuration that finished 3 of them on every
    # instrument (RESULTS 6, "folds 3" three times). A plan of 4 that finishes
    # 3 is not "one fold less": a quarter of the delivered rows then carry no
    # out-of-fold cell at all, so the EM that picks the per-item weights sees
    # 6,209 rows instead of 8,279, the arm screen reads 2,069 instead of
    # 2,759, the calibration slice is 1,600 instead of 2,400, and the widest
    # fold -- the one the screen is read from -- gets a quarter of the fold
    # budget instead of a third. Five losses for one gain, and only the gain
    # was measured, because on a fixture the shortfall is small enough for
    # TabPFN to cover it.
    #
    # So the number of blocks is what a pass can always finish, and the clock
    # that is left buys *passes* instead: a second cover with a fresh
    # partition, whose cells are averaged into the first. Coverage never drops
    # below whole, and extra clock becomes variance reduction rather than a
    # gamble on finishing.
    # v10. 3, measured — and this is a question that could not be asked
    # before the coverage repair, because until then changing K changed the
    # coverage with it ((K-1)/K) and the knob could not be isolated. With
    # coverage pinned at 1.000 for every K, the four ceiling-bearing fixtures
    # answer it cleanly and the answer is monotone:
    #
    #     fixture        K=3       K=4       K=6
    #     fit_cell       0.7639    0.7642    0.7579
    #     hard_unhcr     0.7988    0.7985    0.7894
    #     real_unhcr     0.8394    0.8383    0.8358
    #     sharp_unhcr    0.7641    0.7618    0.7530
    #
    # The clock is fixed, so a larger K gives every fold less of it and every
    # fold model is truncated further; fitting on 83% of the rows instead of
    # 75% does not pay for that. K=3 wins three of four and wins by the most
    # on `sharp_unhcr`, which is the ruler that decides a clock-allocation
    # question ([[simulacrabench-fit-cell-alone-is-not-enough]]).
    "folds": 3,
    "fold_passes": 3,
    # The first fold builds every screen arm and the rest build a third of
    # them, and it is the fold the arm test is read from -- a truncated one
    # does not cost a fold, it costs the roster. An even share is the wrong
    # share for it.
    # v10, measured and then set to the identity. The reasoning was right --
    # a truncated first fold costs the roster, not a fold -- and the fix was
    # wrong: the stop rule sizes the next fold from what the last one cost, so
    # a first fold 35% wider raises the estimate and pushes the *last* fold
    # out of the budget. Measured on four ceiling-bearing fixtures at once:
    # -0.0110 (fit_cell), -0.0148 (hard), -0.0091 (real), -0.0192 (sharp),
    # every one of them finishing 40-90 s early because a fold went missing.
    # The knob stays because the concern is real; the answer is not to buy the
    # first fold's width out of the last fold's budget.
    "wide_fold_scale": 1.0,
    "backoff_depths": [1, 2, 3, 4, 5, 6, 7],
    # v6. How the stratifying columns are ordered. "marginal" is v5: each
    # column's own mutual information with the item. "conditional" is forward
    # selection -- what a column adds given the cells already cut -- which is
    # the criterion matched to what a backoff does with the ranking.
    "backoff_rank": "conditional",
    "backoff_rank_depth": 5,
    "backoff_rank_rows": 6000,
    "mi_bias_correct": True,
    # v6. The same choice for the table `gate_models` fits. Separate from
    # `backoff_rank` so the two can be measured apart -- they are the same
    # idea but they are not the same table, and the gate one is the one
    # UNHCR's documented coding defect lives in.
    "gate_rank": "conditional",
    "backoff_alpha": 2.0,
    # v7. (columns, damping) per member. The one inductive bias the roster did
    # not carry: a product of one-column tables instead of a table over
    # several columns. Empty disables the family.
    "logadd": [[5, 1.0], [17, 0.5]],
    "ridge_grid": [0.003, 0.03],
    "trunk_widths": [8, 64, 256],
    "trunk_ridges": [0.01, 0.05],
    # v9. Two families the clock could never afford to ask about, now that it
    # can. Both are arms: they are built in the wide first fold, and the
    # out-of-fold likelihood on the delivered rows says whether they stay.
    #   `deep`  -- a trunk with a second hidden layer. See torch_members.
    #   `trunk_wide` -- the same family at width 256, which is the capacity
    #                   direction the roster has never been swept in (8 and 64
    #                   were chosen when the whole block had to fit a minute).
    "deep_widths": [64],
    "deep_layers": 2,
    "deep_ridges": [0.01, 0.05],
    "deep_grid": [64],
    "trunk_wide_widths": [256],
    "steps": 400,
    "lr": 0.08,
    "dropout": 0.1,
    "latent_classes": [8, 32, 64],
    # v9. The same family fitted on the joint likelihood of the whole response
    # vector instead of each item's own. Both objectives were implemented in
    # v4 and only one was ever built, because a member costs width and nothing
    # could measure width. The joint one is the interesting half on this
    # instrument: 155 of UNHCR's 213 targets are gated on another target and
    # 73 hang off `s3_11`, so a respondent's answering type is identified by
    # the other items far better than by any one item on its own. The marginal
    # objective optimises the scored quantity directly and cannot use that.
    # Empty disables the family; the screen decides whether it stays.
    "latent_joint_classes": [32, 64],
    "latent_joint_grid": [32, 64],
    "latent_objective": "marginal",
    "latent_steps": 400,
    "latent_lr": 0.08,
    "latent_ridge": 0.03,
    "latent_jitter": 0.5,
    "em_iterations": 200,
    "em_prior": 1.0,
    # v5. How many cells' worth the instrument-wide mixture is worth as the
    # prior on an item's own mixture. Per-item weights over a hundred members
    # are estimated from one item's out-of-fold cells, and that estimate is
    # noisy enough that a wider roster measured *worse* on the harder fixture
    # even though every added member was, individually, defensible. Shrinking
    # each item towards the shared vector is the standard fix and is what
    # makes roster width close to free. 0 restores v4's flat prior.
    "em_pool": 0.0,
    # v7. And how much, chosen on held-out rows rather than set here. After
    # the arm test the roster is about seventy members and an item has 8,279
    # out-of-fold cells, so each weight rests on roughly a hundred of them --
    # thin enough that shrinking towards the instrument-wide mixture is the
    # textbook repair, and thin enough that how far to shrink is not something
    # to guess. 0.0 is in the grid, so the answer can be "do not".
    "em_pool_grid": [0.0, 20.0, 80.0],
    "smoothing": 0.5,
    "gate_smoothing": 0.25,
    "soft_gate": True,
    # v4. The declared gate is a two-cell table in v2: P(sentinel | opened) and
    # P(sentinel | not opened), one pair per item for the whole sample. That is
    # exactly the assumption the UNHCR file breaks -- in five multi-select
    # families the sentinel appears on round-2 rows only, so P(sentinel | not
    # opened) is 0 on one wave and ~1 on the other and the single number is
    # wrong on both. `gate_cond_depth` stratifies that table down the GIVEN
    # columns that carry mutual information with the sentinel itself.
    "gate_cond": True,
    # v5. The same table fitted on every item the same routing event marks,
    # which is what buys the third and fourth stratifying column. 0 disables.
    # v5. Measured neutral-to-slightly-negative on all three ceiling-bearing
    # fixtures (RESULTS §3.3) and it costs 23 members of roster width, which
    # is not free. The argument for it stands -- 59 items read five routing
    # events -- but the fixture cannot see past gateC's depth 2, so this is
    # off until something can measure it. The code stays.
    "gate_group": False,
    "gate_group_depth": 4,
    # v5. The routing event as a tree instead of a chain of columns. One fit
    # per group -- 27 of them on UNHCR -- on the group's per-respondent
    # sentinel share, so it costs seconds. Empty disables.
    # v5. Off: measured -0.0065 on `hard_unhcr` (RESULTS §3.3). A tree beat a
    # column chain for the answer and lost to it for the routing event, which
    # is worth remembering rather than assuming either way next time.
    "gate_tree": {},
    "gate_tree_params": {"max_iter": 120, "max_leaf_nodes": 31,
                         "learning_rate": 0.08, "l2_regularization": 1.0,
                         "min_samples_leaf": 40},
    # v7. `+gateG` and `+gateT` are on the roster to be measured, not to be
    # applied to all 36 base members: the arm test needs enough of them to
    # have an answer and the clock does not want the rest. Prefixes, so a
    # family that is added later joins by name.
    "gate_extra_bases": ("marginal", "backoff", "gbm", "hub", "trunk64"),
    "gate_cond_depth": 2,
    "gate_cond_alpha": 4.0,
    # v4. Nearest neighbours over the one-hot GIVEN block: a local, entirely
    # non-parametric conditional, which is the one inductive bias the roster
    # did not carry. 0 members disables it.
    "knn_k": [1, 5, 15, 40, 300],
    # v4. One exponent per item on the pooled distribution, fitted on
    # held-out cells. A linear pool sits between its members and is therefore
    # systematically flatter than any of them; this measures how much, instead
    # of assuming it either way. 1.0 is in the grid, so it can decline.
    "calibrate": True,
    "calib_rows": 2400,
    "calib_grid": [0.8, 0.9, 0.95, 1.0, 1.06, 1.12, 1.2, 1.35, 1.55, 1.8,
                   2.1, 2.5, 3.0],
    # v7. The damping on the fitted exponent is no longer a constant. It was
    # set when the exponent was fitted on 600 rows; it is now fitted on
    # three times as many, and how much to trust it is a question the
    # held-out half of those rows can answer. 0.75 stays in the grid.
    "calib_shrink": 0.75,
    "calib_shrink_grid": [0.5, 0.75, 1.0],
    "calib_prior": 3.0,
    # v4, off by default. Position of the row inside its block, binned and
    # offered as an extra GIVEN column. See DESIGN.md "row order".
    "order_features": False,
    "order_bins": 24,
    # v6, off by default, and a judgement call rather than a measurement --
    # see DESIGN.md "row order". This is the member-level form of the same
    # idea: instead of changing what every estimator sees, it adds two
    # neighbours-in-file-order estimators to the roster, so the per-item
    # out-of-fold likelihood decides whether row position carries anything.
    # Off means the roster does not contain them at all.
    "pos_members": True,
    "pos_windows": [120, 800],
    "pos_alpha": 4.0,
    # v5. Boosted trees over the raw GIVEN codes. The roster's other
    # families are linear in the one-hot block (logistic, trunk, latent) or
    # stratify a hand-ranked two or three columns at a time (backoff, knn);
    # none of them searches for the interaction that matters. 17 categorical
    # columns, two of them with 46 and 16 levels, is exactly the shape a
    # histogram-boosted tree is good at. sklearn ships in the image.
    # Capacity swept in both directions on held-out cells of `real_unhcr`
    # (RESULTS 3.4). The whole curve spans 0.828-0.832 -- flat -- but every
    # shallow, well-penalised setting beat every deep one, and 300 trees of 31
    # leaves was the *worst* arm on the grid. These trees are variance-limited,
    # not bias-limited, which is also why more clock buys nothing here.
    # v9. Two entries. The second is the same capacity fitted with column
    # subsampling and a per-build seed: v6 reasoned that the bag averages four
    # near-identical tree fits (same `max_features`, same `random_state`,
    # differing only by the third of the rows each fold holds out) and that
    # decorrelating them is the textbook answer for a variance-limited
    # learner, then shipped it off because nothing could measure it. The
    # container can. It costs one more tree block in the wide first fold, and
    # nothing at all after that if the screen drops it.
    "gbm": [{"max_iter": 60, "max_leaf_nodes": 8, "learning_rate": 0.1,
             "l2_regularization": 1.0, "min_samples_leaf": 60},
            {"max_iter": 60, "max_leaf_nodes": 8, "learning_rate": 0.1,
             "l2_regularization": 1.0, "min_samples_leaf": 60,
             "_max_features": 0.7, "_decorrelate": True}],
    # v6. The bag already averages four fits of every member -- one per fold
    # plus the refit on everything -- but every one of those tree fits used
    # max_features=1.0 and the same random_state, so they differ only by the
    # third of the rows each fold holds out. Nearly-identical members average
    # to nearly one member. RESULTS 3.4 measured these trees to be
    # variance-limited rather than bias-limited, and averaging decorrelated
    # fits is the textbook answer to exactly that; the capacity sweep that
    # concluded "more clock buys nothing" was a bias sweep and does not speak
    # to it. Column subsampling plus a per-build seed turns a bag of clones
    # into a real one, at no extra fitting cost. 1.0 is the identity, so the
    # fixtures can decline it.
    "gbm_max_features": 1.0,
    "gbm_decorrelate": False,
    "gbm_hurdle": True,
    "gbm_share": 0.75,
    # v7. The calibration stage measured at 0.9-1.4 s once the pooled vectors
    # were kept instead of restacked, and this was reserving sixty seconds
    # for it out of the refit -- which is the one build fitted on every
    # visible row.
    "calib_reserve": 0.03,
    "gbm_floor": 0.004,
    # v5. Which slice of the wall clock each instrument gets. UNICEF and World
    # Bank are saturated -- every team on the board is inside 0.01 of every
    # other on both -- and UNHCR carries all of the spread, so an equal three
    # way split spends most of the budget where no score can move. Shares are
    # keyed by instrument, not by call order, because the call order is the
    # harness's business and may be one container per instrument.
    # v9. Summing to one. The old vector summed to 0.96, which allocated
    # nothing to nobody, and the shares are now entitlements rather than
    # caps -- see the block in predict() -- so what an instrument leaves
    # goes to whoever has not been called yet.
    # v10. Measured, not assumed. More clock is worth +0.0003 on UNHCR at
    # double (RESULTS 7c, and the fold count is already 4/4 with 8 s of
    # slack), against +0.0013 on UNICEF at 2.5x and +0.0028 on World Bank at
    # 3.3x. The Grand Prize is a plain mean of the three skills, so a nat on
    # a UNICEF cell is worth 5.98 of a nat on a UNHCR cell -- 17,856 scored
    # cells against 132,273 -- and the split that spends 72% of the window on
    # the instrument with the lowest per-cell leverage was never measured,
    # only reasoned from "UNHCR carries all the spread on the leaderboard".
    # That reasoning is about the *displayed* number, which is rounded to
    # 0.01; the final phase reports the exact skill (config.yml, phase 2:
    # `noised: false`, `round_to: null`), and there a sub-0.01 gain on two
    # instruments outweighs a visible one on the third.
    # Moved conservatively, because the deadline is what stops UNHCR's fold
    # loop and dropping it below four folds would cost more than this buys.
    # v10. Reverted, and this is the measurement that reverted it.
    #
    # The case for moving clock to UNICEF and World Bank was good on its face:
    # the Grand Prize is a plain mean of three skills, so a nat on a UNICEF
    # cell is worth 5.98 of a nat on a UNHCR cell (17,856 scored cells against
    # 132,273), the final phase reports the exact skill rather than a rounded
    # one, and more clock measured +0.0013 on UNICEF at 2.5x and +0.0028 on
    # World Bank at 3.3x against a recorded +0.0003 on UNHCR at double.
    #
    # That +0.0003 was measured on `fit_cell` and nowhere else. Taking UNHCR
    # from 0.72 to 0.60 costs **-0.0113 on `sharp_unhcr`** -- the fixture
    # whose conditional is closest to the regime the delivered instrument is
    # actually in -- while `fit_cell` reports +0.0001 for the same change.
    # Four times the gain, in the opposite direction, on the better ruler.
    #
    # Second time a conclusion drawn from `fit_cell` alone has been wrong.
    # A clock trade is judged on the sharpest fixture available, not the
    # most convenient one.
    "budget_share": {"unhcr": 0.72, "world_bank": 0.18, "unicef": 0.10},
    # v5. What each instrument does differently. The group gate answers a flaw
    # the UNHCR documentation records and nothing says the other two share it,
    # and a wide roster costs the same clock everywhere -- so the two saturated
    # instruments run the narrower one and spend what they save on folds.
    # Merged in predict() before the roster is built; a `--set` ablation is
    # applied after this block and therefore still wins.
    "per_instrument": {
        # World Bank is the instrument the linear families own -- the stack
        # gives `trunk64` and its hurdle variants the top weights on every
        # item -- and it is also the one whose GIVEN block is small enough (9
        # columns, 36 pairs) that a large share of the pairs can go in. It is
        # where the pair blocks measured most: +0.0133 on `fitw_mix` and
        # +0.0026 on `fitw_add` at 16, still climbing at 24.
        # v10. 36 is every pair of World Bank's nine GIVEN columns. v6 left
        # this at 16 with "still climbing at 24" and the follow-up was
        # cancelled because a leaderboard submission did not move -- which
        # bounds the change below 0.01 and says nothing else. Measured on both
        # World Bank fixtures at once: +0.0010 (`fitw_add`, the ruler v6
        # certified) and +0.0009 (`fitw_mix`), at no cost in seconds.
        "world_bank": {"calib_rows": 1200, "interaction_pairs": 36},
        # UNICEF has seven GIVEN columns whose full cross is 14,400 cells
        # against 14,885 training rows, so the backoff can go nearly saturated
        # before it runs out of data -- and measured +0.0023 on a ceiling
        # fixture when it does, the only knob on this instrument that moved at
        # all. It is also the instrument that decides the top of the board:
        # ranks 1-2 hold UNICEF 0.52 and ranks 3-4 hold 0.51, all four at
        # UNHCR 0.92.
        # 21 is every pair of UNICEF's seven columns: +0.0006 on `fitu_mix`,
        # whose floor (0.2616) matches the delivered instrument's crowd floor
        # (0.26) more closely than any fixture in this repo matches anything.
        # v11. Twelve items, so the TabPFN ensemble and the whole 14,885-row
        # context are affordable here (83 s call, 4 builds). Measured on
        # `fitu_mix`, paired: stack 0.5762 -> 0.5771; the single estimator
        # alone reads 0.5764, so the stack keeps it and adds to it.
        "unicef": {"calib_rows": 1400,
                   "backoff_depths": [1, 2, 3, 4, 5, 6, 7],
                   "backoff_rank_depth": 7,
                   "interaction_pairs": 21,
                   "tabpfn_estimators": 4,
                   "tabpfn_estimators_refit": 4,
                   "tabpfn_estimators_refit_answer": 4,
                   "tabpfn_rows": 20000},
        # The tiny `sample` schema. UNHCR keys as "unhcr", which has no entry,
        # so it takes the global `calib_rows`.
        "other": {"calib_rows": 600},
    },
    # v5. A second calibration stage for gated items: a slope and an offset on
    # the logit of P(sentinel), fitted on the same held-out cells as the
    # exponent. The exponent moves the whole vector; most of an UNHCR cell's
    # information is the single number that says whether the cell is blank at
    # all, and that number deserves its own two parameters.
    "calib_gate": True,
    "calib_gate_slopes": [0.85, 1.0, 1.15, 1.35, 1.6, 2.0, 2.5, 3.0],
    # v7. Wider on the negative side. 59 of the gated items carry the
    # sentinel on round-2 rows only, so their P(sentinel | wave 1) is zero and
    # not small; an offset grid that stops at -0.3 cannot say so, and the
    # slope alone can only sharpen a logit that is already on the right side.
    "calib_gate_offsets": [-1.2, -0.6, -0.3, -0.12, 0.0, 0.12, 0.3, 0.6],
    # v6. The grader does not score the vector we return: score.py's
    # `floored()` mixes it with a flat vector first, so every entry it ever
    # takes a log of is at least `floor` (0.001 in config.yml). Our own
    # objectives -- the EM that picks the member weights, the exponent grid,
    # the sentinel slope grid -- all took log(max(p, 1e-12)) instead, which
    # prices a confident miss at up to 27.6 nats where the grader prices it at
    # 6.9. On a near-deterministic instrument that is not a rounding detail:
    # it is a systematic tax on sharpness, paid in exactly the regime UNHCR
    # lives in. Flooring the *members* is equivalent to flooring the pool,
    # because the floor is affine and the pool is a convex combination, so EM
    # stays exactly EM -- over the predictive distributions the grader reads.
    "grader_floor": 0.001,
    "floor_aware": True,
    # v6, off by default until measured. How many GIVEN column pairs enter the
    # design matrix as hashed one-hot blocks, for the linear families.
    # v10. 36, measured on the four ceiling-bearing UNHCR fixtures at the
    # same moment as the coverage repair: +0.0001, +0.0009, +0.0001, +0.0000,
    # non-negative on every one and at no cost in seconds (604-605 s against
    # the same 604-605 s). v6 stopped at 8 with "+0.0004 at 32" and read that
    # as flat; it is small, it is also free, and the final phase does not
    # round it away.
    "interaction_pairs": 36,
    "interaction_bins": 48,
    "count_features": True,
    "count_ceiling": 30,
    "min_count": 8,
    "min_torch_rows": 50,
    "min_fold_rows": 120,
    "phase1_budget": 900.0,
    "phase2_budget": 3600.0,
    # Merged into CONFIG once the frame shows this is the final phase, which
    # carries four times the budget and DEV as extra training rows.
    #
    # This was empty through v7, on the reasoning that a setting which has
    # never been scored is not the thing to put on the one November run. That
    # reasoning had it backwards: leaving it empty means the run that decides
    # the prize is the *first* execution of the phase-2 path, and it spends a
    # fifth of the clock it is given. Measured on a phase-2 dry run
    # (`eval_local.py --phase 2`), which v7 was the first version to do at all.
    #
    # `tabpfn` is the reason this block exists. It is the strongest single
    # member on the roster and it costs 205 s, which development cannot pay
    # for and the final can. `folds` 4 and the wider calibration slice are the
    # other two things the extra clock buys directly.
    # v10. Same change as above, one phase up: 4 blocks is inside what v9
    # measured finishing (6), so a pass completes even if the TEST frame is
    # larger than the dry run, and the slack v9 recorded (918 s on UNHCR)
    # buys passes instead of a fold count that has to be guessed right.
    # And 2 passes rather than 3 up here: the bag the shipped vectors come
    # from is the fold models plus the one refit on every visible row, at
    # `bag_full_weight` each. More passes is more fold models diluting the
    # only build that saw all the rows -- 3 x 3 puts the refit at a tenth of
    # the bag against a quarter at v8's 3 x 1. In development the passes are
    # unreachable anyway (UNHCR has 84 s left over and a fold costs 123 s), so
    # they are an outlet for a faster container; in the final there is room,
    # and the dilution is not something any fixture has been asked about yet.
    # `bag_full_weight` is the knob that answers it and it has never moved.
    # v10. 6 -> 4. Phase 1 measures K monotone-decreasing in skill and 6 the
    # worst of three, and shipping the worst-measured arm into the one run
    # that decides the prize is not a defensible default just because a dry
    # run fitted it. 4 is inside the range phase 1 measured and well inside
    # what the phase-2 dry run finished (6 blocks x 2 passes = 12 folds with
    # 300 s of slack), so the passes take up what the smaller K leaves.
    # Still to do before November: the same three-arm sweep at `--phase 2`,
    # where the clock is four times larger and the trade may turn over.
    "phase2_overrides": {"folds": 4, "fold_passes": 3, "calib_rows": 3600},
    "seed": 0,
    "sharpen": [],
    "hurdle": True,
    "instruments": 3,
    # v9. 0.85 was covering two different things at once: overshoot past a
    # deadline, and the container prologue that the old origin did not
    # count. The prologue is now measured (`_process_started_at`), so this
    # only has to cover overshoot -- and overshoot has never been observed
    # positive: every instrument finishes *under* its deadline because the
    # fold loop refuses to start a fold it cannot also refit after. The
    # remaining 7% is 63 s in development and 252 s in the final, against a
    # largest indivisible unit (one item's tree fit, twenty-five optimiser
    # steps) measured in single seconds.
    "safety": 0.93,
    "bag_full_weight": 1.0,
    # v9. A ceiling on the fold loop, kept because the refit is the only
    # build fitted on every visible row and must not be squeezed. It was
    # binding before the refit-aware stop rule got a chance to speak: on
    # UNHCR 0.66 of the call is 377 s and three folds cost 311 s, so a
    # fourth was refused by the ceiling and not by the arithmetic that
    # knows what a refit costs. Raised so that rule decides.
    # v10. Derived, not guessed, and this is the whole of the coverage bug.
    #
    # A fold costs `fold_share x window / K` when the members fill the slice
    # they are handed -- and since v9 they do, because TabPFN and the four new
    # arms saturate it. The stop rule then asks for `cost x (1 + refit_scale)`
    # of room before starting the last fold, and a constant 0.85 never leaves
    # it: the reservation has to scale as `refit_scale / K` and it did not.
    # So the last planned fold was refused *every time*, at every K, and
    # coverage was (K-1)/K:
    #
    #     K = 3  ->  2 folds, coverage 0.667      (v10's first attempt)
    #     K = 4  ->  3 folds, coverage 0.750      (v9, as shipped)
    #     K = 6  ->  5 folds, coverage 0.833
    #
    # v8 finished 3 of 3 only because its roster was cheap enough not to fill
    # the slice; the moment v9 made the folds expensive the same constant
    # started costing a quarter of the delivered rows. That is v9's regression,
    # and lowering K made it worse rather than better.
    #
    # "auto" solves for the reservation instead: K folds plus one refit fit
    # the room exactly when fold_share = 1/(1 + refit_scale/K). Then coverage
    # is 1.000 at any K, and K goes back to being what it should be -- how many
    # rows each fold model is fitted on -- rather than a bet on the clock.
    # v10. Two numbers, and between them they are the coverage bug.
    #
    # `fold_share` carves the refit's budget out of the window; the folds
    # divide what is left. The bug was never the size of that reservation --
    # it was that the fold schedule then spends the fold budget *exactly*.
    # Slice f is `(fold_end - now) / (folds - f)`, so K slices sum to exactly
    # `fold_end`, and the loop refuses to start a fold when `now + cost >
    # fold_end`. Every member checks its deadline between items and therefore
    # overshoots it a little, so after K-1 folds the clock is a few seconds
    # past K-1 slices -- and the last fold is refused by that few seconds.
    # Every time, at every K. Coverage was (K-1)/K and nothing announced it.
    #
    # `fold_slack` takes 5% off each slice so the schedule lands strictly
    # inside its budget and the last fold is never the one that pays for the
    # overshoot of the ones before it.
    "fold_share": 0.80,
    "fold_slack": 0.95,
    "refit_scale": 1.15,

    # ------------------------------------------------------------- v7 ------
    # The ruler moves inside the container.
    #
    # Every fixture this repo owns mis-states the regime. The delivered UNHCR
    # file has a crowd-marginal floor of 0.48 and a leaderboard top of 0.92;
    # no generator we have reaches a ceiling above 0.79 at that floor, and
    # `fit_cell` reaches its ceiling by drawing an independent Dirichlet per
    # GIVEN cell -- a target with nothing to generalise across cells, which is
    # the opposite of a real questionnaire. `headroom.py` on it returns a
    # shortfall that is flat across all 213 items (top item 2.2% of the gap,
    # gated 0.0373 and ungated 0.0334 nats per item): the signature of the
    # fixture's own generator, not of anything we could fix.
    #
    # So the families whose value cannot be measured locally are measured
    # *here*, on the delivered rows, out of fold, against a cluster-robust t.
    # Two thousand held-out respondents x 213 items is three orders of
    # magnitude more evidence than one leaderboard day, it costs no submission,
    # and it is measured on the instrument being predicted rather than on a
    # proxy. The competition rules point the same way: development feedback
    # may not be used to "exploit repeated score feedback", and an out-of-fold
    # split of the visible rows is not feedback at all.
    # Every optional family is *built*, and then removed from the mixture at
    # the out-of-fold likelihood matrix rather than rebuilt, so eight
    # questions cost nothing beyond the width they already paid for. The
    # arms are all cheap ones on purpose: a tree capacity would have to be
    # refitted to be asked about, and the clock does not have it.
    "screen": True,
    "screen_tstat": 3.0,            # cluster-robust t an arm has to clear
    "screen_min_rows": 1200,        # below this every arm is simply kept
    # The container is killed at 16 GB, and a killed container costs the day's
    # one submission. The widest roster is 198 members read out over 3,381
    # rows and 213 items, which is 2.2 GB of members plus 1.4 GB of
    # likelihood; the guard drops arms, widest first, until the estimate fits.
    # It is a ceiling and not a target: on the shipped sizes nothing is
    # dropped, and it exists so that a bigger phase-2 frame degrades instead
    # of dying.
    # Calibrated against the measurement rather than guessed: a phase-2 UNHCR
    # call whose estimate is 3.31 GB peaked at 7.34 GB of RSS, so the resident
    # set runs about 1.8x the estimate plus the CUDA context. 3.6 keeps the
    # worst case inside half of the 16 GB the grader kills at.
    # v9. The roster is wider (deep, trunk256, the decorrelated tree), so the
    # estimate this is compared against is larger. Recalibrated the same way:
    # a phase-2 estimate of 3.31 GB peaked at 7.34 GB of RSS, so the resident
    # set is about 1.8x the estimate plus the CUDA context. 4.4 keeps the
    # worst case near 9.5 GB, inside the 16 GB the grader kills at, and the
    # guard still sheds the widest arms first if a phase-2 frame is bigger
    # than the dry run.
    "member_budget_gb": 4.4,
    "shed_order": ["gateT", "gateG", "gbm_decorr", "trunk_wide", "deep",
                   "latent_joint", "latent64", "logadd", "backoff_deep",
                   "knn_small", "tabpfn", "regime"],
    # `gateG`/`gateT` are not on this list. They are the only two arms whose
    # width is expensive rather than free -- 54 members and two extra model
    # fits -- and they have now measured clearly negative on every fixture
    # in v5, v6 and v7. Asking again costs the third fold, and the third
    # fold is worth more than the answer. The code stays and the two flags
    # turn them back on.
    "screen_arms": ["knn_small", "pos", "hub", "backoff_deep", "latent64",
                    "logadd", "sharp_tables", "sharp_trees", "tabpfn",
                    "deep", "trunk_wide", "gbm_decorr", "latent_joint",
                    "regime"],
    # v8. TabPFN v2, the one family on the roster that is not fitted here.
    # `models.txt` names the repository so the organisers pre-download the
    # weights before the clock starts; the network is gone by the time this
    # runs, so every lookup is local and finding nothing simply drops the
    # family. Measured at 0.19 s an item over the full 8,279 rows -- 39 s for
    # a build, the same order as the tree block.
    # Off in both phases, and that is a measurement rather than caution.
    #
    # Run properly it is the strongest single member the roster has ever had:
    # `tabpfn+gateC` takes the top per-item weight on four of six panels, up
    # to 0.535 on phase-2 `hard_unhcr`, ahead of every backoff and every tree.
    # And it loses anyway, because what it costs is a fold:
    #
    #     phase 2, TabPFN on / off      folds      skill
    #     hard_unhcr    0.8007 / 0.8082   5 / 6   -0.0075
    #     real_unhcr    0.8403 / 0.8423   5 / 6   -0.0020
    #     fit_cell      0.7661 / 0.7654   5 / 6   +0.0007
    #
    # 837-986 s across the builds, against a budget the organisers have now
    # confirmed is 900 s (3,600 s in the final) for the three instruments
    # together. It wins the mixture and loses the clock, in both phases.
    #
    # The code stays, and so does the arm, because the verdict is about
    # seconds and not about quality: an instrument with fewer items, or a
    # budget that grows, flips it. The dependency does not ship -- a package
    # that can only fail the image build for a family that is switched off is
    # a pure liability. Turning it on means putting `tabpfn` back into
    # `requirements.txt` and the repo into `models.txt`.
    #
    # (It also cost three measurements to find out. The first two recorded a
    # family that had never run: placed behind the tree block, and then behind
    # the torch families, both of which size themselves to the clock that is
    # left and hand whatever follows a deadline already past.)
    # v9. On, and on as an arm: it is built in the wide first fold and the
    # delivered rows say whether it stays. v8's rejection was never about
    # quality -- it took the top per-item weight on four of six panels -- it
    # was that a build cost 60-110 s and there are five of them. Two changes
    # in `tabpfn_member` (the 17 delivered columns instead of 50, and the
    # backbone built once instead of 209 times) put a build at 20 s, which is
    # the same order as the tree block it sits beside. See RESULTS 5.3.
    "tabpfn": True,
    "tabpfn_repo": "Prior-Labs/TabPFN-v2-clf",
    "tabpfn_file": "tabpfn-v2-classifier-v2_default.ckpt",
    "tabpfn_estimators": 1,
    # v11. The refit is the one build fitted on every visible row and it is
    # the build the shipped vectors lean on; TabPFN's in-context ensemble is
    # its main quality axis (measured standalone: 1 -> 4 estimators is
    # +0.025 on sharp_unhcr and +0.019 on fit_cell). Folds keep 1 -- their
    # job is the out-of-fold likelihood, and a fold is paid for four times.
    # v11. Measured in the stack, phase-1 clock, sharp_unhcr: a 4-estimator
    # full pass in the refit is -0.005 at fold_share 0.80 (it truncates the
    # refit's other members) and +0.001 at 0.65 (it truncates the folds). The
    # standalone +0.02 does not survive the bag, which already averages three
    # fold fits on different contexts. 1 in development; the final phase has
    # the clock for more and its dry run decides.
    "tabpfn_estimators_refit": 1,
    "tabpfn_seconds_refit": 240.0,
    # v11. The answer-only half of a hurdle, fitted by TabPFN on the rows the
    # routing actually asked. Measured standalone on the two sharpest
    # fixtures: the K+1-way TabPFN scores 0.7056 / 0.7533 (sharp_unhcr /
    # fit_cell, 8 estimators) and the hurdle-decomposed one 0.7614 / 0.7639
    # -- which is the whole v10 stack's score (0.7623 / 0.7642) from a single
    # member. The stack had hurdle variants of every fitted family except
    # this one. `hurdle:tabpfn` and `hurdleC:tabpfn` join the roster; the
    # answer pass costs about half the full pass (asked rows are a fraction
    # of the visible rows and ungated items reuse the full fit).
    "tabpfn_hurdle": True,
    # The refit's answer pass gets the ensemble; the fold passes stay at 1,
    # because a fold is paid for K times and its job is the out-of-fold
    # likelihood, and the ensemble is worth +0.02 on the shipped member.
    "tabpfn_estimators_refit_answer": 1,
    # v11. Covariate shift between the visible rows and the rows being
    # predicted. Every choice the container makes on held-out cells -- the
    # EM weights, the exponent, the sentinel slope, the pool rule -- is made
    # on out-of-fold *visible* rows and applied to the delivered ones. That
    # is right only if the two blocks are draws from the same population.
    # Nothing in the frame guarantees it: the harness does not shuffle, and
    # the delivered roles were assigned when the file was built. If the
    # held-out block sits in a different region of the GIVEN space (one
    # survey round, one country, one cohort), the weights are tuned to the
    # wrong mix. So the out-of-fold cells are raked to the GIVEN marginals of
    # the rows being predicted, on the columns where the two blocks differ
    # by more than chance (a chi-square at `shift_alpha`, Bonferroni over
    # the GIVEN columns). Under no shift no column passes and this is the
    # identity -- exactly the identity, not approximately.
    "shift_weights": True,
    "shift_alpha": 1e-3,
    "shift_clip": [0.2, 5.0],
    "shift_iterations": 20,
    # Every visible row, capped. 0.096 s an item at 3,500 rows against 0.168 s
    # at all 8,279 -- 20 s a build against 35 s -- and measured on the two
    # ceiling-bearing fixtures the wider context is worth +0.0029 (`fit_cell`)
    # and +0.0020 (`sharp_unhcr`) at the same fold count. Context is this
    # model's main axis and 63 s is what UNHCR had left over after the third
    # fold, so this is exactly what the reclaimed clock buys.
    # The cap is not 0 ("all") because phase 2 hands over a much larger frame
    # and row attention does not stay linear; 9,000 is every development row
    # and a ceiling on the final.
    "tabpfn_rows": 9000,
    "tabpfn_count_features": False,
    "tabpfn_max_width": 10,      # 209 of UNHCR's 213 items
    "tabpfn_floor": 0.004,
    # A ceiling on the family, not a target: 35 s a build is the measurement,
    # and this is what stops a slow container from spending a fold on it.
    # Whatever it does not reach keeps the crowd, which the stack weights out.
    "tabpfn_seconds": 70.0,
    # v7. The smoothing constants were set against fixtures whose conditionals
    # are Dirichlet draws. The delivered instrument is not in that regime: the
    # crowd marginal scores 0.48 on it and the board's leader scores 0.92,
    # which only happens when a large share of cells is nearly determined by
    # the GIVEN block. On such a cell `backoff_alpha` 2.0 caps a 20-row cell
    # at 20/22 and charges 0.095 nats for a cell that costs nothing to get
    # right, and a per-item exponent cannot undo it -- reaching 0.999 from
    # 0.909 needs an exponent that would wreck every undetermined cell.
    # The repair belongs in the members: put a sharp table and a sharp tree on
    # the roster next to the smooth ones and let the out-of-fold likelihood
    # say, per item, which regime that item is in. Both are free -- the same
    # counts, the same tree fit, a different prior on top.
    # v7. How the members are combined, chosen per item on held-out cells.
    # A linear pool sits between its members and is therefore flatter than any
    # of them; the fitted exponent corrects that with one number for the whole
    # item, which is the wrong shape for the correction -- the flattening is
    # large where the members disagree and nil where they agree, and that
    # varies cell by cell, not item by item. A log pool is exactly the
    # cell-adaptive version of the same correction. Both go in the grid with
    # the exponent, and "linear" can win.
    "pool_rule": "auto",
    # v10. The regime family. See `regime_member` for why it exists; the
    # short version is that the delivered instrument's two endpoints (crowd
    # 0.48, leader 0.92) only close if most cells are settled by the GIVEN
    # block, and in that regime every counting family on the roster shrinks
    # those cells towards the one distribution that is guaranteed to be wrong
    # for them -- the pooled marginal, which is the mixture of the patterns
    # and is sharp on none of them.
    #
    # `regime_classes` is how many answer patterns an item is allowed. 2 is
    # the number the arithmetic points at (a pooled entropy of log 2 = 0.69
    # nats against the 0.656 the observed floor implies); 4 is there because
    # that arithmetic is an average over 213 items and some of them will have
    # more. It is an arm: built in the wide first fold, and the delivered
    # rows decide whether it stays.
    # v10, off. The family was built for the regime the two leaderboard
    # endpoints imply -- most cells settled by the GIVEN block, where
    # shrinking a cell towards the pooled marginal is the one target
    # guaranteed to be wrong for it -- and then measured at exactly 0.0000 on
    # five fixtures, including two built to have precisely that structure
    # (`rg2_*`, floor 0.499 / ceiling 0.891). The reason is that v7 already
    # fixed it: `backoff@auto` fits alpha per level per item by leave-one-out,
    # and an alpha of 0.1 puts a settled twenty-row cell at 0.995 against this
    # family's 0.998. The deterministic-cell tax was not still being paid.
    # Neutral is not free -- twelve members of width in the wide first fold,
    # which is the fold whose truncation costs the roster -- so it ships off.
    # The code and the arm stay; `[2, 4]` turns it back on.
    "regime_classes": [],
    "regime_depths": [2, 3],
    "regime_kappa": 4.0,
    "regime_max_cells": 4_000_000,
    "regime_grid": [2, 4],
    "backoff_alphas": [2.0, "auto"],
    "backoff_sharp_alphas": [2.0, "auto"],
    "backoff_alpha_grid": [0.1, 0.3, 0.7, 1.5, 3.0, 6.0],
    "gbm_floors": [0.004, 0.0004],
    "gbm_sharp_floors": [0.004, 0.0004],
    # v7. The hub family: marginalise an item over another PREDICT item.
    #
    # 155 of UNHCR's 213 targets are gated on another PREDICT item and 73 hang
    # off `s3_11` alone, so most of a cell's information is a question we also
    # have to predict. `gate_refine` already propagates a predicted parent,
    # but only into the sentinel slot, only through the declared gate, and
    # only as "opened or not". This is the same idea without those three
    # restrictions: P(y | x) = sum_h P(h | x) T(y | h, x), with h any
    # high-information PREDICT item, the table over the answer as well as the
    # sentinel, and ungated items included. The marginalisation is exact.
    "hub": True,
    "hub_candidates": 6,
    "hub_depth": 2,
    "hub_alpha": 2.0,
    "hub_max_width": 8,
    # v7. Candidate capacities the screen picks between. Index 0 is v6's.
    "gbm_grid": [
        {"max_iter": 60, "max_leaf_nodes": 8, "learning_rate": 0.1,
         "l2_regularization": 1.0, "min_samples_leaf": 60},
        {"max_iter": 200, "max_leaf_nodes": 24, "learning_rate": 0.06,
         "l2_regularization": 1.0, "min_samples_leaf": 25},
        {"max_iter": 40, "max_leaf_nodes": 6, "learning_rate": 0.12,
         "l2_regularization": 2.0, "min_samples_leaf": 120},
        {"max_iter": 60, "max_leaf_nodes": 8, "learning_rate": 0.1,
         "l2_regularization": 1.0, "min_samples_leaf": 60,
         "_max_features": 0.7, "_decorrelate": True},
    ],
    "knn_small_k": [1, 5, 15],
    "backoff_deep_depths": [6, 7],
    "latent64_classes": [64],
    "logadd_grid": [[5, 1.0], [17, 0.5]],
}
# Local sweeps (SB_CONFIG) and `package.py --set` both land in _OVERRIDE, and
# predict() applies it again after the per-instrument block so that an
# ablation arm always wins over a per-instrument default.
_OVERRIDE = json.loads(os.environ.get("SB_CONFIG", "{}"))
CONFIG.update(_OVERRIDE)
_BASE = None

_EPS = 1e-12
_TABPFN_PATH = None
_STAGE = {}
_BAG = {}


# ------------------------------------------------------------- instrument --

class Instrument:
    """Everything about one call that is read off the schema and the frame."""

    def __init__(self, frame, schema):
        records = schema["items"]
        self.items = [n for n, r in records.items()
                      if r["class"] in ("GIVEN", "PREDICT")]
        self.given = [n for n in self.items if records[n]["class"] == "GIVEN"]
        self.targets = [n for n in self.items
                        if records[n]["class"] == "PREDICT"]
        sentinel = schema["gated_value"]

        # Option order is the schema's, sentinel last -- never the data's.
        self.cats = {}
        self.gate = {}
        for name in self.items:
            values = list(records[name]["values"])
            gate = records[name].get("gate")
            self.cats[name] = values + ([sentinel] if gate else [])
            self.gate[name] = gate

        self.n = len(frame)
        self.code = {}
        self.isna = {}
        for name in self.items:
            column = frame[name]
            lookup = {value: i for i, value in enumerate(self.cats[name])}
            codes = column.map(lookup)
            self.isna[name] = codes.isna().to_numpy()
            self.code[name] = codes.fillna(-1).astype(np.int64).to_numpy()

        # A respondent is visible when every scored cell of theirs is there.
        blank = np.zeros(self.n, bool)
        for name in self.targets:
            blank |= self.isna[name]
        self.fit_rows = np.flatnonzero(~blank)
        self.out_rows = np.flatnonzero(blank)

        # v5. Which instrument this call is, so the budget can be split by
        # where the leaderboard actually moves rather than three ways. Keyed
        # off the schema's shape, which is fixed and does not depend on the
        # dataset description staying worded the way it is today.
        scored = len(self.targets)
        self.key = ("unhcr" if scored >= 150 else
                    "world_bank" if scored >= 40 else
                    "unicef" if scored >= 8 else "other")

        split = schema.get("split", {})
        n_train = int(split.get("n_train", 0))
        n_dev = int(split.get("n_dev", 0))
        # Phase 2 delivers TRAIN and DEV complete, so the visible count says
        # which phase this is, and therefore what the budget is.
        self.phase = 2 if (n_train and len(self.fit_rows)
                           > n_train + 0.5 * n_dev) else 1

        # v4, off by default. The delivered frame is the visible rows in file
        # order followed by the held-out rows in file order, so a row's
        # position inside its own block is a covariate the harness hands to
        # every submission. Offered as one more GIVEN column, which puts it in
        # front of the design matrix, the backoff stratifier and the gate
        # models at once, and leaves the stack free to ignore it. See
        # DESIGN.md, "row order", for why this is a judgement call.
        if CONFIG["order_features"]:
            bins = max(2, int(CONFIG["order_bins"]))
            position = np.zeros(self.n)
            for block in (self.fit_rows, self.out_rows):
                if block.size:
                    position[block] = np.arange(block.size) / block.size
            name = "__position__"
            self.items.append(name)
            self.given.append(name)
            self.cats[name] = ["b%02d" % i for i in range(bins)]
            self.gate[name] = None
            self.code[name] = np.minimum((position * bins).astype(np.int64),
                                         bins - 1)
            self.isna[name] = np.zeros(self.n, bool)

        # Where the row sits inside its own block, in [0, 1). score.py builds
        # the frame as the visible rows in file order followed by the held-out
        # rows in file order, so this is a covariate the harness hands to every
        # submission alike. Computed always, read only by `pos_members`.
        self.position = np.zeros(self.n)
        for block in (self.fit_rows, self.out_rows):
            if block.size:
                self.position[block] = np.arange(block.size) / block.size

        self.width = {name: len(self.cats[name]) for name in self.items}
        self.groups = {}
        for j, name in enumerate(self.targets):
            self.groups.setdefault(self.width[name], []).append(j)

        # Parents before children, so the gate chain can be walked in one pass.
        self.depth = {}
        for name in self.items:
            steps, seen, walk = 0, set(), name
            while self.gate.get(walk) and walk not in seen:
                seen.add(walk)
                walk = self.gate[walk]["parent"]
                steps += 1
            self.depth[name] = steps

        # The answer-only view: a gated item's sentinel slot is dropped and the
        # rows that carry it are masked out, so P(answer | asked) is fitted on
        # the respondents who were actually asked. See the hurdle members.
        self.answer_width = {n: self.width[n] - (1 if self.gate[n] else 0)
                             for n in self.items}
        self.answer_groups = {}
        for j, name in enumerate(self.targets):
            self.answer_groups.setdefault(
                self.answer_width[name], []).append(j)

    def target_codes(self):
        return np.stack([self.code[n] for n in self.targets], axis=1)


COUNT_LEVEL = re.compile(r"^\s*(\d+)\s*(?:[-+]\s*\d*\s*)?\+?\s*$")


def counted(values):
    """The levels as numbers, when the column is a count of something.

    A count column's levels are bare numbers or bare ranges -- "0", "3",
    "6-10", "11+" -- and small. That is what tells `s2_3_boys_0_5` (children in
    the household) apart from `s2_1_hh_relationship_lbl` ("2. Spouse/partner",
    whose leading digit is a code, not a quantity) and from `user.age`
    ("45+", a band whose sum with anything means nothing). Returns None when
    the column is not one.
    """
    numbers = []
    for value in values:
        found = COUNT_LEVEL.match(str(value))
        if not found:
            return None
        numbers.append(float(found.group(1)))
    if not numbers or max(numbers) > CONFIG["count_ceiling"]:
        return None
    return np.asarray(numbers, np.float32)


def count_blocks(inst):
    """Totals over the count columns that belong together.

    Skip logic keys off quantities the GIVEN block reports in pieces: whether a
    household contains a child under five is `s2_3_boys_0_5 + s2_3_girls_0_5`,
    which one-hot columns can only approximate and a ridge-penalised model will
    not push to a decision. Columns whose names differ in exactly one token are
    parts of one total, so `s2_3_*` gives the household size and `*_0_5`,
    `*_6_17`, `*_18_59`, `*_60plus` give its four age bands.
    """
    values, groups = {}, {}
    for name in inst.given:
        numbers = counted(inst.cats[name])
        if numbers is None:
            continue
        codes = inst.code[name]
        values[name] = np.where(codes < 0, 0.0, numbers[np.maximum(codes, 0)])
        tokens = name.split("_")
        keys = [tokens[0] if len(tokens) < 3 else "_".join(tokens[:2])]
        if len(tokens) > 3:
            keys.append("*_" + "_".join(tokens[3:]))
        for key in keys:
            groups.setdefault(key, []).append(name)

    blocks = []
    for name, column in values.items():
        scale = max(column.max(), 1.0)
        blocks.append((column / scale)[:, None])
        blocks.append((column > 0).astype(np.float32)[:, None])
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        total = sum(values[name] for name in members)
        scale = max(total.max(), 1.0)
        blocks.append((total / scale)[:, None])
        blocks.append((total > 0).astype(np.float32)[:, None])
        spread = sum((values[name] > 0).astype(np.float32) for name in members)
        blocks.append((spread / len(members))[:, None])
    return blocks


def interaction_pairs(inst):
    """One-hot blocks for the GIVEN column pairs, hashed to a fixed width.

    The design matrix is additive: the linear families on the roster
    (logistic, trunk, latent) see each column on its own and can only reach a
    joint split through the trunk's one hidden layer. The trees reach them
    directly, which is what v5 measured, and the backoff now reaches them
    because `rank_given` picks its columns conditionally -- this is the same
    repair for the third family.

    Pairs are chosen once for the instrument, by summed bias-corrected mutual
    information of the pair's joint code with the items, minus what the two
    columns already carry apart -- so a pair earns its place only by what it
    adds over the additive terms already in the matrix. Hashed to
    `interaction_bins` so a 46 x 16 pair costs the same as a 2 x 2 one.
    """
    if not CONFIG["interaction_pairs"] or len(inst.given) < 2:
        return []
    cached = getattr(inst, "_pairs", None)
    if cached is not None:
        return cached
    bins = int(CONFIG["interaction_bins"])
    rows = inst.fit_rows
    cap = int(CONFIG["backoff_rank_rows"])
    if 0 < cap < len(rows):
        rows = rows[np.random.default_rng(CONFIG["seed"]).choice(
            len(rows), cap, replace=False)]
    targets = inst.targets[::max(1, len(inst.targets) // 40)]
    scores = []
    for a, b in [(a, b) for i, a in enumerate(inst.given)
                 for b in inst.given[i + 1:]]:
        ca, cb = np.maximum(inst.code[a][rows], 0), np.maximum(
            inst.code[b][rows], 0)
        joint = ca * inst.width[b] + cb
        joint, inverse = np.unique(joint, return_inverse=True)
        joint, jc = inverse.astype(np.int64), joint.size
        if jc < 4:
            continue
        gain = 0.0
        for name in targets:
            y = np.maximum(inst.code[name][rows], 0)
            yc = inst.width[name]
            both = mutual_information(joint, y, jc, yc) \
                - max(jc - 1, 0) * max(yc - 1, 0) / (2.0 * len(rows))
            apart = max(
                mutual_information(ca, y, inst.width[a], yc)
                - max(inst.width[a] - 1, 0) * max(yc - 1, 0) / (2.0 * len(rows)),
                mutual_information(cb, y, inst.width[b], yc)
                - max(inst.width[b] - 1, 0) * max(yc - 1, 0) / (2.0 * len(rows)))
            gain += max(both - apart, 0.0)
        scores.append((gain / max(len(targets), 1), a, b))
    scores.sort(reverse=True)
    blocks = []
    for _, a, b in scores[:int(CONFIG["interaction_pairs"])]:
        code = (np.maximum(inst.code[a], 0) * inst.width[b]
                + np.maximum(inst.code[b], 0)) % bins
        onehot = np.zeros((inst.n, bins), np.float32)
        onehot[np.arange(inst.n), code] = 1.0
        blocks.append(onehot)
    inst._pairs = blocks
    return blocks


def design(inst):
    """One-hot every GIVEN column, plus its ordinal position, plus a bias.

    An unseen or absent value gets its own slot rather than falling silently
    into the first category. Count columns additionally contribute their value,
    an is-any indicator, and the totals of the groups they belong to.
    """
    blocks = []
    for name in inst.given:
        card = inst.width[name]
        codes = inst.code[name].copy()
        codes[codes < 0] = card                       # unseen / missing
        onehot = np.zeros((inst.n, card + 1), np.float32)
        onehot[np.arange(inst.n), codes] = 1.0
        blocks.append(onehot)
        if card > 2:
            ordinal = np.where(inst.code[name] < 0, 0.0,
                               inst.code[name] / max(card - 1, 1)) - 0.5
            blocks.append(ordinal.astype(np.float32)[:, None])
    if CONFIG["count_features"]:
        blocks.extend(count_blocks(inst))
    for pair in interaction_pairs(inst):
        blocks.append(pair)
    blocks.append(np.ones((inst.n, 1), np.float32))
    return np.concatenate(blocks, axis=1).astype(np.float32) if blocks else \
        np.ones((inst.n, 1), np.float32)


# ------------------------------------------------------------- estimators --

def marginal_member(inst, fit_rows, rows):
    counts = []
    for name in inst.targets:
        y = inst.code[name][fit_rows]
        table = np.bincount(y[y >= 0], minlength=inst.width[name]).astype(
            np.float64) + CONFIG["smoothing"]
        counts.append(np.tile(table / table.sum(), (len(rows), 1)))
    return counts


def mutual_information(x, y, nx, ny):
    both = (x >= 0) & (y >= 0)
    x, y = x[both], y[both]
    if x.size < CONFIG["min_count"]:
        return 0.0
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny) / x.size
    px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())


def conditional_mi(key, kc, g, gc, y, yc, correct=True):
    """MI(item ; column | the cells already stratified), in nats.

    The plain ranking asks what a column says about the answer *on its own*.
    That is the wrong question for a backoff, which will read the columns
    together: a column that matters only in combination with one already
    chosen carries almost no marginal information and is ranked last, so the
    chain never reaches it. This asks what a column adds given the cells the
    chain has already cut, which is the quantity the next level actually
    spends its rows on.

    `correct` subtracts the Miller-Madow bias, (rows-1)(cols-1)/2N per cell.
    Without it the estimate rewards cardinality for its own sake -- UNHCR's
    governorate column has 46 levels against 8,279 rows, so an independent
    column of that width still scores about 0.005 nats, which is the same
    size as the real signal it is being ranked against.
    """
    counts = np.bincount(key * (gc * yc) + g * yc + y,
                         minlength=kc * gc * yc).reshape(kc, gc, yc)
    total = counts.sum(axis=(1, 2)).astype(np.float64)
    keep = total >= CONFIG["min_count"]
    if not keep.any():
        return 0.0
    counts = counts[keep].astype(np.float64)
    total = total[keep]
    joint = counts / total[:, None, None]
    pg = joint.sum(axis=2, keepdims=True)
    py = joint.sum(axis=1, keepdims=True)
    nz = joint > 0
    mi = np.zeros(len(total))
    product = pg * py
    ratio = np.zeros_like(joint)
    ratio[nz] = joint[nz] / product[nz]
    mi = (joint * np.where(nz, np.log(np.maximum(ratio, _EPS)), 0.0)).sum(
        axis=(1, 2))
    if correct:
        seen_g = (counts.sum(axis=2) > 0).sum(axis=1)
        seen_y = (counts.sum(axis=1) > 0).sum(axis=1)
        mi = mi - np.maximum(seen_g - 1, 0) * np.maximum(seen_y - 1, 0) \
            / (2.0 * total)
    weight = total / total.sum()
    return float((weight * mi).sum())


def rank_given(inst, fit_rows):
    """Per item, the GIVEN columns in the order a backoff should read them.

    `backoff_rank` "marginal" is v5: sort once by each column's own mutual
    information with the item. "conditional" picks the columns one at a time,
    each time by what the column adds given the cells already chosen, which is
    forward selection against the estimator that will use the ranking.
    """
    order = {}
    card = {g: inst.width[g] for g in inst.given}
    seen = {g: np.maximum(inst.code[g][fit_rows], 0) for g in inst.given}
    correct = bool(CONFIG["mi_bias_correct"])
    if CONFIG["backoff_rank"] != "conditional":
        for name in inst.targets:
            y = inst.code[name][fit_rows]
            order[name] = sorted(
                inst.given, reverse=True,
                key=lambda g: mutual_information(seen[g], y, card[g],
                                                 inst.width[name])
                - (correct * max(card[g] - 1, 0)
                   * max(inst.width[name] - 1, 0)
                   / (2.0 * max(len(fit_rows), 1))))
        return order

    # Forward selection. Only the first `backoff_rank_depth` positions are
    # chosen this way -- they are the only ones a backoff of that depth ever
    # reads -- and the tail keeps the marginal order so the list stays a full
    # permutation of the GIVEN block.
    limit = max(1, int(CONFIG["backoff_rank_depth"]))
    sample = np.arange(len(fit_rows))
    cap = int(CONFIG["backoff_rank_rows"])
    if 0 < cap < len(sample):
        sample = np.random.default_rng(CONFIG["seed"]).choice(
            len(sample), cap, replace=False)
    for name in inst.targets:
        y_all = inst.code[name][fit_rows]
        y = np.maximum(y_all[sample], 0)
        yc = inst.width[name]
        chosen, rest = [], list(inst.given)
        key = np.zeros(len(sample), np.int64)
        kc = 1
        for _ in range(min(limit, len(rest))):
            best, score = None, -np.inf
            for g in rest:
                value = conditional_mi(key, kc, seen[g][sample], card[g],
                                       y, yc, correct)
                if value > score:
                    best, score = g, value
            chosen.append(best)
            rest.remove(best)
            key = key * card[best] + seen[best][sample]
            key, inverse = np.unique(key, return_inverse=True)
            key, kc = inverse.astype(np.int64), key.size
        tail = sorted(
            rest, reverse=True,
            key=lambda g: mutual_information(seen[g], y_all, card[g], yc))
        order[name] = chosen + tail
    return order


def backoff_member(inst, fit_rows, rows, order, depth, alpha=None):
    """Dirichlet backoff from the marginal down `depth` stratifying columns.

    `alpha` "auto" fits the strength of the prior per item and per level of the
    chain, by leave-one-out on the rows the table was counted from. That
    matters more than it looks. A fixed alpha of 2 caps a twenty-row cell at
    20/22 and charges 0.095 nats for a cell the GIVEN block determines
    outright, and this instrument has many of those -- the crowd marginal
    scores 0.48 on it and the board's leader 0.92, which is only reachable if
    a large share of cells is nearly settled by the block. A fixed alpha of
    0.4 gets those right and pays for it on the cells that are genuinely
    uncertain. Which cells an item has is not something to be assumed either
    way, and leave-one-out on the delivered rows is the cheapest possible way
    to ask: the counts are already built, and the answer is a grid search over
    six numbers on a vector that is already in cache.
    """
    auto = isinstance(alpha, str)
    fixed = CONFIG["backoff_alpha"] if alpha is None else \
        (0.0 if auto else float(alpha))
    grid = np.asarray(CONFIG["backoff_alpha_grid"], float) if auto else None
    out = []
    for name in inst.targets:
        width = inst.width[name]
        y = inst.code[name][fit_rows]
        keep = y >= 0
        y_fit = y[keep]
        table = np.bincount(y_fit, minlength=width).astype(np.float64) \
            + CONFIG["smoothing"]
        crowd = table / table.sum()
        running = np.tile(crowd, (len(rows), 1))
        inner = np.tile(crowd, (y_fit.size, 1)) if auto else None
        seat = np.arange(y_fit.size)
        key_fit = np.zeros(y_fit.size, np.int64)
        key_out = np.zeros(len(rows), np.int64)
        size = 1
        for column in order[name][:depth]:
            card = inst.width[column]
            key_fit = key_fit * card + np.maximum(
                inst.code[column][fit_rows][keep], 0)
            key_out = key_out * card + np.maximum(inst.code[column][rows], 0)
            size *= card
            table = np.bincount(key_fit * width + y_fit,
                                minlength=size * width).astype(np.float64)
            table = table.reshape(size, width)
            step = fixed
            if auto and y_fit.size > 4:
                own = table[key_fit, y_fit] - 1.0
                total = table.sum(axis=1)[key_fit] - 1.0
                prior = inner[seat, y_fit]
                best = -np.inf
                for candidate in grid:
                    score = np.log(np.maximum(
                        (own + candidate * prior) / (total + candidate),
                        _EPS)).sum()
                    if score > best:
                        best, step = score, float(candidate)
                rows_fit = table[key_fit]
                inner = ((rows_fit + step * inner)
                         / (rows_fit.sum(axis=1, keepdims=True) + step))
            counts = table[np.minimum(key_out, size - 1)]
            running = ((counts + step * running)
                       / (counts.sum(axis=1, keepdims=True) + step))
        out.append(running)
    return out


def _regime_fit(table, classes, seed, iterations=40, prior=0.5):
    """A finite mixture of multinomials over the cells of one level.

    `table` is (cells x options) of counts. Returns (theta, pi) -- `classes`
    answer distributions and their weights -- or None if the level has too few
    populated cells to say anything.
    """
    populated = np.flatnonzero(table.sum(axis=1) > 0)
    if populated.size <= classes * 2 or table.shape[1] < 2:
        return None
    counts = table[populated]
    width = counts.shape[1]
    # Deterministic init: the `classes` heaviest cells, whose profiles are the
    # ones estimated well enough to be worth starting from, smoothed so no
    # component starts with a structural zero it can never leave.
    heavy = populated[np.argsort(-table[populated].sum(axis=1))[:classes]]
    theta = table[heavy] + prior
    theta /= theta.sum(axis=1, keepdims=True)
    weights = np.full(classes, 1.0 / classes)
    logs = np.log(np.maximum(theta, _EPS))
    for _ in range(iterations):
        score = counts @ logs.T + np.log(np.maximum(weights, _EPS))
        score -= score.max(axis=1, keepdims=True)
        resp = np.exp(score)
        resp /= np.maximum(resp.sum(axis=1, keepdims=True), _EPS)
        weights = resp.mean(axis=0)
        theta = resp.T @ counts + prior
        theta /= theta.sum(axis=1, keepdims=True)
        logs = np.log(np.maximum(theta, _EPS))
    return theta, weights


def regime_member(inst, fit_rows, rows, order, depth, classes, alpha=None,
                  kappa=None):
    """A backoff whose prior is the cell's own regime, not the pooled marginal.

    This is the one family the roster never had, and the regime the delivered
    instrument is in is what asks for it.

    Read the two endpoints the leaderboard has been reporting since the first
    week. The crowd marginal scores 0.48 on UNHCR and the leader scores 0.92,
    which in nats is a pooled entropy of 0.656 and a conditional entropy of
    0.101: the GIVEN block removes 85% of the uncertainty. A questionnaire
    does that by *routing* -- an item has a handful of answer patterns and the
    block picks between them -- and the arithmetic only closes if a large
    share of cells is nearly settled.

    In that regime shrinking a cell towards the pooled marginal is not a
    conservative default, it is the worst available target. The marginal is
    the mixture of the patterns, so it sits between them and is sharp on none
    of them, and `backoff_alpha` 2.0 caps a twenty-row settled cell at 20/22 =
    0.909 and charges 0.095 nats for a cell that costs nothing to get right.
    Our whole distance from the board's leader is 0.025 nats a cell.

    `backoff@auto` answers that by lowering alpha, which buys the settled
    cells and pays for it on the uncertain ones, because it is one number per
    level. The repair belongs in the target instead: fit the level's cells as
    a mixture of `classes` answer distributions and shrink each cell towards
    the one it looks like. A settled cell then goes to 0.998 without touching
    the uncertain cells at all, because they are shrunk towards a different
    component.

    It is safe where it is uninformative. A cell with no counts gets the
    mixture's own weights back, whose average is the pooled marginal, so the
    member degrades exactly to `backoff`; and `kappa` blends the regime prior
    with the level above by how many rows the cell actually has, so a cell
    that cannot identify its component does not pretend to.
    """
    fixed = CONFIG["backoff_alpha"] if alpha is None else float(alpha)
    kappa = float(CONFIG["regime_kappa"] if kappa is None else kappa)
    out = []
    for name in inst.targets:
        width = inst.width[name]
        y = inst.code[name][fit_rows]
        keep = y >= 0
        y_fit = y[keep]
        table = np.bincount(y_fit, minlength=width).astype(np.float64) \
            + CONFIG["smoothing"]
        crowd = table / table.sum()
        running = np.tile(crowd, (len(rows), 1))
        key_fit = np.zeros(y_fit.size, np.int64)
        key_out = np.zeros(len(rows), np.int64)
        size = 1
        for column in order[name][:depth]:
            card = inst.width[column]
            key_fit = key_fit * card + np.maximum(
                inst.code[column][fit_rows][keep], 0)
            key_out = key_out * card + np.maximum(inst.code[column][rows], 0)
            size *= card
            if size * width > CONFIG["regime_max_cells"]:
                break
            table = np.bincount(key_fit * width + y_fit,
                                minlength=size * width).astype(np.float64)
            table = table.reshape(size, width)
            fitted = _regime_fit(table, int(classes), CONFIG["seed"])
            counts = table[np.minimum(key_out, size - 1)]
            if fitted is None:
                prior = running
            else:
                theta, weights = fitted
                logs = np.log(np.maximum(theta, _EPS))
                score = table @ logs.T + np.log(np.maximum(weights, _EPS))
                score -= score.max(axis=1, keepdims=True)
                resp = np.exp(score)
                resp /= np.maximum(resp.sum(axis=1, keepdims=True), _EPS)
                cell_prior = resp @ theta
                # How far to believe the cell's own regime, by how many rows
                # it has to say it with. An empty cell falls all the way back
                # to the level above, which is what `backoff` would have done.
                total = table.sum(axis=1)[np.minimum(key_out, size - 1)]
                trust = (total / (total + kappa))[:, None]
                prior = trust * cell_prior[np.minimum(key_out, size - 1)] \
                    + (1.0 - trust) * running
            running = ((counts + fixed * prior)
                       / (counts.sum(axis=1, keepdims=True) + fixed))
        out.append(running)
    return out


def logadd_member(inst, fit_rows, rows, order, blocks, damping):
    """A product of one-column tables, not a table over several columns.

    Every counting family on the roster estimates a *joint*: `backoff` cuts
    cells by x1, then x1 x2, then x1 x2 x3, and backs off when a cell runs
    thin. That is the right shape when the truth really is a lookup over those
    columns together, and it is the expensive shape otherwise -- the cell count
    multiplies while the rows do not.

    Real survey conditionals are usually not that. They are several partly
    independent pushes -- where the household lives, which round asked, how
    many children it has -- each moving the answer a little, which is additive
    in log-odds rather than a joint cell. Estimating that as a joint spends the
    sample on cells the truth does not distinguish; estimating it as a sum of
    one-column tables costs `sum of cardinalities` parameters instead of their
    product.

    So: each column's own table, smoothed towards the item's marginal, entered
    as its log-odds against that marginal.

        log p(y | x) = log p0(y) + damping * SUM_c [log P_c(y | x_c) - log p0(y)]

    `damping` 1.0 is naive Bayes, which double counts whatever the columns
    share; below 1 it shrinks that back. Which value -- and whether the family
    is worth anything on this item at all -- is the per-item stack's business,
    not a thing to assume here.
    """
    alpha = CONFIG["backoff_alpha"]
    out = []
    for name in inst.targets:
        width = inst.width[name]
        y = inst.code[name][fit_rows]
        keep = y >= 0
        y_fit = y[keep]
        table = np.bincount(y_fit, minlength=width).astype(np.float64) \
            + CONFIG["smoothing"]
        p0 = table / table.sum()
        base = np.log(p0)
        logits = np.tile(base, (len(rows), 1))
        for column in order[name][:blocks]:
            card = inst.width[column]
            codes_fit = np.maximum(inst.code[column][fit_rows][keep], 0)
            counts = np.bincount(codes_fit * width + y_fit,
                                 minlength=card * width).astype(np.float64)
            counts = counts.reshape(card, width)
            conditional = (counts + alpha * p0[None, :]) / (
                counts.sum(axis=1, keepdims=True) + alpha)
            shift = np.log(conditional) - base[None, :]
            logits = logits + damping * shift[
                np.maximum(inst.code[column][rows], 0)]
        logits -= logits.max(axis=1, keepdims=True)
        vector = np.exp(logits)
        out.append(vector / vector.sum(axis=1, keepdims=True))
    return out


def torch_members(inst, X, fit_rows, rows, hidden, ridges, deadline,
                  skip=False, steps=None, lr=None, dropout=None,
                  answer_only=False, layers=1):
    """Batched multinomial regressions, one shared design matrix, GPU if there.

    `hidden` None fits an independent linear model per item. An integer fits a
    shared trunk of that width first: a low-rank respondent code, which is what
    a latent-trait / IRT model is, and it lets an item borrow strength from
    every other item. `skip` keeps a direct linear path alongside the trunk.

    Every head carries its own bias, started at the item's log marginal, so the
    optimiser begins from the crowd and spends its steps on the conditioning
    rather than on rediscovering the marginal.

    `answer_only` drops each gated item's sentinel slot and masks out the rows
    that carry it, so what comes back is P(answer | asked) fitted on the
    respondents who were asked -- the answer half of a hurdle model, whose gate
    half `gate_refine` supplies.

    One member comes back per entry of `ridges`; they are fitted together as one
    batched problem, which is nearly free next to fitting them in turn.
    """
    import torch

    steps = CONFIG["steps"] if steps is None else steps
    lr = CONFIG["lr"] if lr is None else lr
    dropout = CONFIG["dropout"] if dropout is None else dropout

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(CONFIG["seed"])
    dtype = torch.float32

    Xf = torch.as_tensor(X[fit_rows], dtype=dtype, device=device)
    Xo = torch.as_tensor(X[rows], dtype=dtype, device=device)
    Y = inst.target_codes()
    n, d = Xf.shape
    L = len(ridges)
    lam = torch.as_tensor(ridges, dtype=dtype, device=device)
    n_items = len(inst.targets)

    # Items are grouped by option count, so each group is one dense tensor.
    layout = inst.answer_groups if answer_only else inst.groups
    grouped, params, heads, biases, linears = [], [], [], [], []
    for width, members in sorted(layout.items()):
        idx = np.asarray(members)
        codes = Y[fit_rows][:, idx]
        if answer_only:
            # A sentinel codes as this item's last class; those rows carry no
            # answer, so they are masked out and their target is parked at 0.
            asked = codes < width
            codes = np.where(asked, codes, 0)
        else:
            asked = np.ones(codes.shape, bool)
        yg = torch.as_tensor(codes, dtype=torch.long, device=device)
        mask = torch.as_tensor(asked.astype(np.float32), device=device)
        grouped.append((width, idx, yg, mask))

        start = []
        for position, j in enumerate(idx):
            keep = codes[:, position][asked[:, position]]
            table = np.bincount(keep, minlength=width).astype(np.float64) \
                + CONFIG["smoothing"]
            start.append(np.log(np.maximum(table, _EPS)))
        start = np.stack(start)
        start -= start.mean(axis=1, keepdims=True)
        bias = torch.as_tensor(np.tile(start, (L, 1, 1)), dtype=dtype,
                               device=device).requires_grad_(True)
        biases.append(bias)
        params.append(bias)

    fan = d if hidden is None else hidden
    # v9. `layers` > 1 puts hidden layers between the trunk and the heads.
    # The roster's parametric families were a linear model (`logistic`), a
    # rank-h linear model with one ReLU (`trunk`) and a mixture whose gate is
    # linear (`latent`); a second hidden layer is the smallest thing that is
    # none of those, and on a GPU it costs milliseconds. Whether it earns its
    # width is the container's question, not one to settle here.
    stack = []
    if hidden is None:
        trunk = None
    else:
        trunk = torch.empty(L, d, hidden, device=device, dtype=dtype)
        torch.nn.init.normal_(trunk, std=(1.0 / max(d, 1)) ** 0.5)
        trunk.requires_grad_(True)
        params.append(trunk)
        for _ in range(max(int(layers) - 1, 0)):
            layer = torch.empty(L, hidden, hidden, device=device, dtype=dtype)
            torch.nn.init.normal_(layer, std=(1.0 / max(hidden, 1)) ** 0.5)
            layer.requires_grad_(True)
            stack.append(layer)
            params.append(layer)

    for width, idx, _, _ in grouped:
        head = torch.zeros(L, fan, len(idx), width, device=device, dtype=dtype,
                           requires_grad=True)
        heads.append(head)
        params.append(head)
        if hidden is not None and skip:
            linear = torch.zeros(L, d, len(idx), width, device=device,
                                 dtype=dtype, requires_grad=True)
            linears.append(linear)
            params.append(linear)

    opt = torch.optim.Adam(params, lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    drop = torch.nn.Dropout(dropout) if hidden is not None else None

    def forward(source, train):
        if trunk is None:
            code = source.unsqueeze(0).expand(L, source.shape[0], d)
        else:
            code = torch.relu(torch.einsum("nd,ldh->lnh", source, trunk))
            if train and dropout > 0:
                code = drop(code)
            for layer in stack:
                code = torch.relu(torch.einsum("lnh,lhg->lng", code, layer))
                if train and dropout > 0:
                    code = drop(code)
        out = []
        for position, ((width, idx, _, _), head) in enumerate(
                zip(grouped, heads)):
            logits = torch.einsum("lnh,lhgk->lngk", code, head) \
                + biases[position].unsqueeze(1)
            if linears:
                logits = logits + torch.einsum("nd,ldgk->lngk", source,
                                               linears[position])
            out.append(logits)
        return out

    for step in range(steps):
        if step % 25 == 0 and time.time() > deadline:
            break
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for (width, idx, yg, mask), logits in zip(grouped, forward(Xf, True)):
            per = torch.nn.functional.cross_entropy(
                logits.reshape(-1, width),
                yg.unsqueeze(0).expand(L, n, len(idx)).reshape(-1),
                reduction="none").view(L, n, len(idx)) * mask
            # Mean over the rows that count for each item, not over all rows.
            per = per.sum(dim=1) / mask.sum(dim=0).clamp_min(1.0)
            loss = loss + per.sum(dim=1) / n_items
        penalty = sum(head.pow(2).sum(dim=(1, 2, 3)) for head in heads)
        for linear in linears:
            penalty = penalty + linear.pow(2).sum(dim=(1, 2, 3))
        if trunk is not None:
            penalty = penalty + trunk.pow(2).sum(dim=(1, 2))
        for layer in stack:
            penalty = penalty + layer.pow(2).sum(dim=(1, 2))
        (loss + lam * penalty / n_items).sum().backward()
        opt.step()
        schedule.step()

    out = [[None] * n_items for _ in range(L)]
    with torch.no_grad():
        for (width, idx, _, _), logits in zip(grouped, forward(Xo, False)):
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
            for position, j in enumerate(idx):
                for level in range(L):
                    out[level][j] = probs[level, :, position, :].astype(
                        np.float64)
    del Xf, Xo, params, heads, biases, linears, stack
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def latent_members(inst, X, fit_rows, rows, classes, deadline,
                   objective=None, steps=None, lr=None, ridge=None):
    """Latent-class members: a respondent belongs to one of C answering types.

    P(class | GIVEN) is a softmax of the design matrix and P(answer | class) is
    a free table per item, so the conditional is a mixture rather than a single
    softmax -- which is the shape a latent trait actually induces, and the
    structure `torch_measure` is in the image for. It is fitted by gradient
    descent on the marginal likelihood, either of the whole response vector
    ("joint", which lets an item's class be identified by all the other items)
    or of each item on its own ("marginal", which optimises the scored quantity
    directly). Both are offered; the stack decides which was right.

    One member per entry of `classes`.
    """
    import torch

    objective = CONFIG["latent_objective"] if objective is None else objective
    steps = CONFIG["latent_steps"] if steps is None else steps
    lr = CONFIG["latent_lr"] if lr is None else lr
    ridge = CONFIG["latent_ridge"] if ridge is None else ridge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    Xf = torch.as_tensor(X[fit_rows], dtype=dtype, device=device)
    Xo = torch.as_tensor(X[rows], dtype=dtype, device=device)
    Y = inst.target_codes()
    n, d = Xf.shape
    n_items = len(inst.targets)

    grouped = []
    for width, members in sorted(inst.groups.items()):
        idx = np.asarray(members)
        yg = torch.as_tensor(Y[fit_rows][:, idx], dtype=torch.long,
                             device=device)
        start = np.stack([
            np.log(np.maximum(np.bincount(
                np.maximum(inst.code[inst.targets[j]][fit_rows], 0),
                minlength=width).astype(np.float64) + CONFIG["smoothing"],
                _EPS)) for j in idx])
        start -= start.mean(axis=1, keepdims=True)
        flat = (torch.arange(len(idx), device=device) * width).view(1, -1) + yg
        grouped.append((width, idx, flat.reshape(-1), start))

    members = []
    for n_classes in classes:
        torch.manual_seed(CONFIG["seed"])
        gate = torch.zeros(d, n_classes, device=device, dtype=dtype,
                           requires_grad=True)
        thetas, params = [], [gate]
        for width, idx, _, start in grouped:
            theta = torch.as_tensor(
                np.tile(start, (n_classes, 1, 1)), dtype=dtype, device=device)
            theta = theta + CONFIG["latent_jitter"] * torch.randn(
                theta.shape, device=device, dtype=dtype)
            theta.requires_grad_(True)
            thetas.append(theta)
            params.append(theta)

        opt = torch.optim.Adam(params, lr=lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        for step in range(steps):
            if step % 20 == 0 and time.time() > deadline:
                break
            opt.zero_grad(set_to_none=True)
            log_gate = torch.log_softmax(Xf @ gate, dim=1)          # [n, C]
            if objective == "joint":
                acc = torch.zeros(n, n_classes, device=device, dtype=dtype)
                for (width, idx, flat, _), theta in zip(grouped, thetas):
                    table = torch.log_softmax(theta, dim=-1).reshape(
                        n_classes, -1)
                    acc = acc + table[:, flat].view(
                        n_classes, n, len(idx)).sum(dim=2).t()
                loss = -torch.logsumexp(log_gate + acc, dim=1).mean() / n_items
            else:
                weight = log_gate.exp()
                loss = 0.0
                for (width, idx, flat, _), theta in zip(grouped, thetas):
                    table = torch.softmax(theta, dim=-1).reshape(n_classes, -1)
                    picked = table[:, flat].view(n_classes, n, len(idx))
                    marginal = torch.einsum("nc,cng->ng", weight, picked)
                    loss = loss - torch.log(marginal + 1e-9).mean(
                        dim=0).sum() / n_items
            (loss + ridge * gate.pow(2).sum() / n_items).backward()
            opt.step()
            schedule.step()

        member = [None] * n_items
        with torch.no_grad():
            weight = torch.softmax(Xo @ gate, dim=1)
            for (width, idx, _, _), theta in zip(grouped, thetas):
                table = torch.softmax(theta, dim=-1)                # [C, G, K]
                probs = torch.einsum("nc,cgk->ngk", weight, table)
                probs = probs.float().cpu().numpy()
                for position, j in enumerate(idx):
                    member[j] = probs[:, position, :].astype(np.float64)
        members.append(member)
        del gate, thetas, params
        if device == "cuda":
            torch.cuda.empty_cache()
    del Xf, Xo
    return members


def given_codes(inst, counts=None):
    """The GIVEN block as raw integer categories, unseen levels in their own.

    Trees want the codes, not the one-hot: a 46-level governorate is one column
    to split on rather than 47 columns a ridge has to spend penalty on.

    `counts` False drops the derived count columns and returns the seventeen
    delivered ones alone. A tree wants them -- a gate keys off a sum of two
    count columns and a split has to rebuild it -- but they are 33 of the 50
    columns, and for a model whose cost is linear in its feature count and
    which can read the same sum out of the categorical levels itself, they are
    two thirds of the bill for the third of the information. Measured on
    UNHCR: 0.28 s an item at 50 columns against 0.096 s at 17.
    """
    if counts is None:
        counts = CONFIG["count_features"]
    slot = "_given_codes" if counts else "_given_codes_raw"
    cached = getattr(inst, slot, None)
    if cached is not None:
        return cached
    columns = []
    for name in inst.given:
        codes = inst.code[name].astype(np.float64)
        codes[codes < 0] = inst.width[name]          # unseen gets its own level
        columns.append(codes)
    categorical = list(range(len(columns)))
    if counts:
        # The same totals the design matrix carries. A gate keys off "is there
        # a child under five in this household", which is a sum of two count
        # columns; a tree splitting on the two levels separately has to spend
        # depth to rebuild it, and can only approximate it.
        columns += [block[:, 0].astype(np.float64)
                    for block in count_blocks(inst)]
    matrix = np.column_stack(columns) if columns else \
        np.zeros((inst.n, 1), np.float64)
    setattr(inst, slot, (matrix, categorical))
    return getattr(inst, slot)


def gbm_member(inst, fit_rows, rows, params, deadline, answer_only=False):
    """Histogram-boosted trees, one model per item, over the raw GIVEN codes.

    The roster's parametric families are linear in the one-hot block and its
    counting families stratify at most three columns deep, hand-ranked by
    mutual information with the item taken one column at a time. Neither
    searches for a joint split -- "round 2 AND from Aleppo AND living in
    Akkar" -- and a household survey's routing is full of them. Trees do
    nothing else.

    Fitted per item because the items do not share a target space. Cost is
    roughly linear in the number of options, so the budget is spent in schema
    order and whatever is left when `gbm_seconds` runs out keeps the crowd
    marginal: a member is allowed to be partly the crowd, and the per-item
    stack then simply weights those items on the other families.

    `answer_only` drops the sentinel slot and the rows that carry it, which is
    the half a hurdle member multiplies by its own P(asked).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    matrix, categorical = given_codes(inst)
    train, read = matrix[fit_rows], matrix[rows]
    stop = deadline
    # v7. A capacity in `gbm_grid` may carry its own decorrelation, so the two
    # knobs are per-config rather than global: the screen puts the plain and
    # the decorrelated tree on the roster at once and lets the delivered rows
    # say whether averaging decorrelated fits is worth a member of width.
    params = dict(params)
    max_features = params.pop("_max_features", CONFIG["gbm_max_features"])
    decorrelate = params.pop("_decorrelate", CONFIG["gbm_decorrelate"])
    extra = {}
    if max_features < 1.0:
        extra["max_features"] = float(max_features)
    seed = CONFIG["seed"]
    if decorrelate:
        # A function of the rows this build was handed, so it is stable across
        # runs and different for every fold and for the refit.
        seed = int(zlib.crc32(np.ascontiguousarray(fit_rows).tobytes())
                   & 0x7fffffff)
    floors = list(CONFIG.get("gbm_floors") or [CONFIG["gbm_floor"]])
    out = [[] for _ in floors]
    for name in inst.targets:
        width = inst.answer_width[name] if answer_only else inst.width[name]
        y = inst.code[name][fit_rows]
        keep = y >= 0
        if answer_only and inst.gate[name]:
            keep &= y < inst.width[name] - 1
        table = np.bincount(y[keep], minlength=inst.width[name])[:width]
        table = table.astype(np.float64) + CONFIG["smoothing"]
        crowd = table / table.sum()
        if time.time() > stop or keep.sum() < CONFIG["min_count"] * 4:
            for slot in out:
                slot.append(np.tile(crowd, (len(rows), 1)))
            continue
        target = y[keep]
        levels = np.unique(target)
        if levels.size < 2:
            for slot, floor in zip(out, floors):
                vector = np.full(width, floor / max(width - 1, 1))
                vector[levels[0]] = 1.0 - floor
                slot.append(np.tile(vector / vector.sum(), (len(rows), 1)))
            continue
        try:
            model = HistGradientBoostingClassifier(
                categorical_features=categorical, early_stopping=False,
                random_state=seed, **params, **extra)
            model.fit(train[keep], target)
            probability = model.predict_proba(read)
        except Exception:
            for slot in out:
                slot.append(np.tile(crowd, (len(rows), 1)))
            continue
        raw = np.zeros((len(rows), width))
        raw[:, model.classes_.astype(np.int64)] = probability
        for slot, floor in zip(out, floors):
            full = (1.0 - floor) * raw + floor * crowd[None, :]
            slot.append(full / full.sum(axis=1, keepdims=True))
    return out


def tabpfn_checkpoint():
    """Where the pre-downloaded TabPFN v2 classifier weights are, or None.

    `models.txt` names the Hugging Face repository, so the organisers fetch it
    while the image is built -- before the container exists and before the
    clock starts -- and the file is in the ordinary hub cache by the time
    `predict()` runs. The grader then removes the network, so every path here
    is a local one: ask the hub cache without letting it reach out, and if
    that fails, look for the file. Finding nothing is not an error; the family
    simply does not join the roster.
    """
    global _TABPFN_PATH
    if _TABPFN_PATH is not None:
        return _TABPFN_PATH or None
    _TABPFN_PATH = ""
    repo, name = CONFIG["tabpfn_repo"], CONFIG["tabpfn_file"]
    try:
        from huggingface_hub import hf_hub_download
        _TABPFN_PATH = hf_hub_download(repo, name, local_files_only=True)
        return _TABPFN_PATH
    except Exception:
        pass
    import glob
    roots = [os.environ.get("HF_HOME", ""), os.environ.get("HF_HUB_CACHE", ""),
             os.path.expanduser("~/.cache/huggingface")]
    for root in roots:
        if not root:
            continue
        found = glob.glob(os.path.join(root, "**", name), recursive=True)
        if found:
            _TABPFN_PATH = found[0]
            return _TABPFN_PATH
    return None


def torch_cuda():
    """Is there a GPU to put a member on."""
    if not torch_available():
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


_TABPFN_SPEC = None


def tabpfn_backbone(path):
    """The built model, once per process, handed to every per-item classifier.

    `TabPFNClassifier(model_path=<path>)` reconstructs the architecture and
    reloads the state dict on every `fit`, and a build calls it 209 times.
    The library's own answer to that is `ClassifierModelSpecs`: pass the built
    model rather than a path and `initialize_tabpfn_model` returns it
    untouched. Returning None on any failure keeps the slow path working.
    """
    global _TABPFN_SPEC
    if _TABPFN_SPEC is not None:
        return _TABPFN_SPEC or None
    _TABPFN_SPEC = False
    try:
        from tabpfn.base import (initialize_tabpfn_model,
                                 ClassifierModelSpecs)
        models, configs, _, inference = initialize_tabpfn_model(
            model_path=path, which="classifier", fit_mode="fit_preprocessors")
        _TABPFN_SPEC = ClassifierModelSpecs(
            model=models[0], architecture_config=configs[0],
            inference_config=inference)
    except Exception:
        _TABPFN_SPEC = False
    return _TABPFN_SPEC or None


def tabpfn_member(inst, fit_rows, rows, deadline, estimators=None,
                  answer_only=False, reuse=None):
    """TabPFN v2: a transformer that does Bayesian inference in context.

    Every other family on the roster is fitted on this instrument's rows and
    nothing else -- a table, a softmax, a low-rank trunk, a boosted tree. This
    one arrives already trained, on millions of synthetic tabular problems
    drawn from a prior over structural causal models, and "fitting" it is
    putting the visible rows in its context window. That is a different
    inductive bias from anything the roster carries, and on datasets of this
    size and shape -- thousands of rows, seventeen categorical columns -- it is
    the strongest single tabular learner published.

    v4 dropped this idea after measuring a TabICL variant at 95x the cost for
    a tie. That measurement was made on the *deployed* sandbox, where PREDICT
    does not depend on GIVEN at all, so no estimator can be ranked there; the
    rejection never had evidence behind it. Measured properly it costs 0.19 s
    an item at the full 8,279 rows, which is 39 s for a build -- the same
    order as the tree block it sits beside.

    Items wider than `tabpfn_max_width` keep the crowd (the classifier is
    trained for up to ten classes; 209 of UNHCR's 213 items are inside it).
    """
    path = tabpfn_checkpoint()
    if not path:
        return None
    # Only on the GPU. The runtime has an H100 and every measurement here was
    # made on one; on a CPU this family is roughly twenty times slower, which
    # would spend `tabpfn_seconds` a build to return the crowd marginal for
    # most of the items. If the device is not there, the family is not worth
    # the clock -- and the rest of the roster stands on its own.
    if not torch_cuda():
        return None
    from tabpfn import TabPFNClassifier

    # v9. Two changes, and together they are what makes this family
    # affordable: 0.28 s an item became 0.096 s, which is 20 s a build rather
    # than 60-110 s, against a family that was rejected in v8 for costing a
    # fold and nothing else.
    #
    #  (1) The columns. `given_codes` adds 33 derived count columns to the 17
    #      delivered ones, and this model's cost is linear in its feature
    #      count -- it attends over features. A boosted tree needs them (a
    #      gate keys off a sum, and a split has to rebuild it); a transformer
    #      doing in-context inference over categorical levels does not.
    #  (2) The backbone. Every `TabPFNClassifier(model_path=<str>)` rebuilds
    #      the architecture and reloads the state dict, and there are 209 of
    #      them per build. `ClassifierModelSpecs` is the supported way to hand
    #      an already-built model in, so it is built once per call.
    matrix, categorical = given_codes(
        inst, counts=bool(CONFIG["tabpfn_count_features"]))
    spec = tabpfn_backbone(path)
    rng = np.random.default_rng(CONFIG["seed"] + 5)
    take = fit_rows
    cap = int(CONFIG["tabpfn_rows"])
    if 0 < cap < len(take):
        take = take[rng.choice(len(take), cap, replace=False)]
    train_all, read = matrix[take].astype(np.float32), \
        matrix[rows].astype(np.float32)
    floor = float(CONFIG["tabpfn_floor"])
    limit = int(CONFIG["tabpfn_max_width"])
    out = []
    for j, name in enumerate(inst.targets):
        gated = bool(inst.gate[name])
        # v11. The answer-only pass: an ungated item's answer is its whole
        # vector, so the full fit is reused where the caller hands one in.
        if answer_only and not gated and reuse is not None:
            out.append(reuse[j])
            continue
        width = inst.answer_width[name] if answer_only else inst.width[name]
        y = inst.code[name][take]
        keep = y >= 0
        if answer_only and gated:
            keep &= y < inst.width[name] - 1
        table = np.bincount(y[keep], minlength=inst.width[name])[:width]
        table = table.astype(np.float64) + CONFIG["smoothing"]
        crowd = table / table.sum()
        if width > limit or time.time() > deadline \
                or keep.sum() < CONFIG["min_count"] * 4 \
                or np.unique(y[keep]).size < 2:
            out.append(np.tile(crowd, (len(rows), 1)))
            continue
        try:
            model = TabPFNClassifier(
                n_estimators=int(estimators or CONFIG["tabpfn_estimators"]),
                device="cuda" if torch_cuda() else "cpu",
                model_path=spec if spec is not None else path,
                ignore_pretraining_limits=True,
                random_state=CONFIG["seed"],
                categorical_features_indices=list(categorical))
            model.fit(train_all[keep], y[keep])
            probability = model.predict_proba(read)
        except Exception:
            out.append(np.tile(crowd, (len(rows), 1)))
            continue
        full = np.zeros((len(rows), width))
        full[:, np.asarray(model.classes_, np.int64)] = probability
        full = (1.0 - floor) * full + floor * crowd[None, :]
        out.append(full / full.sum(axis=1, keepdims=True))
    return out


def knn_member(inst, fit_rows, rows, k_values, deadline):
    """The crowd, but only the part of it that looks like this respondent.

    Every other family on the roster is parametric in the GIVEN block: a
    softmax, a low-rank trunk, or a backoff down two or three hand-ranked
    columns. This one is none of those. Similarity is the number of GIVEN
    columns two respondents share, the `k` most similar visible rows vote, and
    the vote is backed off to the crowd by the same Dirichlet weight the
    backoff members use.

    It buys the interactions the ranked backoff cannot reach -- a Damascene
    household in Akkar on round 2 is a cell no depth-3 stratification will
    ever name -- without the smoothness a ridge-penalised softmax imposes.
    Ties in the match count are broken arbitrarily by `topk`, which samples
    within the tied shell and is what makes small `k` usable at all.

    Falls back to the crowd marginal when torch is missing or the deadline has
    passed, which keeps the roster and therefore the weights lined up.
    """
    crowd = marginal_member(inst, fit_rows, rows)
    if not k_values:
        return []
    if not torch_available() or time.time() > deadline:
        return [crowd for _ in k_values]
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        blocks = []
        for name in inst.given:
            card = inst.width[name]
            codes = inst.code[name].copy()
            codes[codes < 0] = card
            onehot = np.zeros((inst.n, card + 1), np.float32)
            onehot[np.arange(inst.n), codes] = 1.0
            blocks.append(onehot)
        if not blocks:
            return [crowd for _ in k_values]
        Z = np.concatenate(blocks, axis=1)

        offsets, total = {}, 0
        for name in inst.targets:
            offsets[name] = total
            total += inst.width[name]
        targets = np.zeros((len(fit_rows), total), np.float32)
        seat = np.arange(len(fit_rows))
        for name in inst.targets:
            codes = inst.code[name][fit_rows]
            keep = codes >= 0
            targets[seat[keep], offsets[name] + codes[keep]] = 1.0

        Zf = torch.as_tensor(Z[fit_rows], device=device)
        T = torch.as_tensor(targets, device=device)
        alpha = CONFIG["backoff_alpha"]
        members = []
        for k in k_values:
            k = int(min(k, len(fit_rows)))
            if k < 1:
                members.append(crowd)
                continue
            counted_rows = []
            for start in range(0, len(rows), 1024):
                chunk = rows[start:start + 1024]
                Zo = torch.as_tensor(Z[chunk], device=device)
                sim = Zo @ Zf.t()
                pick = sim.topk(k, dim=1).indices
                vote = torch.zeros_like(sim)
                vote.scatter_(1, pick, 1.0)
                counted_rows.append((vote @ T).cpu().numpy().astype(np.float64))
                del Zo, sim, pick, vote
            counts = np.concatenate(counted_rows, axis=0)
            member = []
            for j, name in enumerate(inst.targets):
                block = counts[:, offsets[name]:offsets[name] + inst.width[name]]
                back = crowd[j]
                member.append((block + alpha * back)
                              / (block.sum(axis=1, keepdims=True) + alpha))
            members.append(member)
            if time.time() > deadline:
                members += [crowd] * (len(k_values) - len(members))
                break
        del Zf, T
        if device == "cuda":
            torch.cuda.empty_cache()
        return members
    except Exception:
        return [crowd for _ in k_values]


def hub_items(inst, fit_rows):
    """The PREDICT items worth marginalising other items over.

    A hub earns its place by how much it says about the rest of the
    instrument, so it is ranked by summed bias-corrected mutual information
    with every other target -- the same statistic `rank_given` uses on the
    GIVEN columns, applied to the block we have to predict. On UNHCR the
    ranking recovers the routing spine without being told it: `s3_11` decides
    the sentinel of 73 items, `s3_15` of 19, `s6_10` of 13.

    Narrow items only. The table a hub is read through is indexed by its value,
    so a 46-level hub would spend the counts it is supposed to sharpen.
    """
    cached = getattr(inst, "_hubs", None)
    if cached is not None:
        return cached
    rows = fit_rows
    cap = int(CONFIG["backoff_rank_rows"])
    if 0 < cap < len(rows):
        rows = rows[np.random.default_rng(CONFIG["seed"]).choice(
            len(rows), cap, replace=False)]
    sample = inst.targets[::max(1, len(inst.targets) // 48)]
    scores = []
    for name in inst.targets:
        width = inst.width[name]
        if width > CONFIG["hub_max_width"] or width < 2:
            continue
        h = np.maximum(inst.code[name][rows], 0)
        total = 0.0
        for other in sample:
            if other == name:
                continue
            y = np.maximum(inst.code[other][rows], 0)
            yc = inst.width[other]
            total += max(mutual_information(h, y, width, yc)
                         - max(width - 1, 0) * max(yc - 1, 0)
                         / (2.0 * len(rows)), 0.0)
        scores.append((total / max(len(sample), 1), name))
    scores.sort(reverse=True)
    inst._hubs = [name for _, name in scores[:int(CONFIG["hub_candidates"])]]
    return inst._hubs


def hub_member(inst, fit_rows, rows, order, base):
    """P(y | x) = sum_h P(h | x) T(y | h, x), h another PREDICT item.

    155 of UNHCR's 213 targets are gated on an item that is itself hidden, and
    73 of them hang off `s3_11` alone, so most of what a cell costs is a
    question we also have to answer. The roster already carries one half of
    that: `gate_refine` walks the declared gate and propagates the parent's
    predicted distribution. But it propagates it only into the sentinel slot,
    only through the one parent the schema names, and only as "opened or not".

    This is the same marginalisation without those three restrictions. The hub
    is whichever narrow PREDICT item carries the most information about this
    item -- the declared parent, usually, but chosen rather than assumed. The
    table is over the answer as well as the sentinel. Ungated items get one
    too. And the sum runs over the hub's values rather than over a binary.

    The sum is exact: no sampling, no chain. `base` supplies P(hub | x) from
    the members already built in this same build, so a fold's hub distribution
    is as out-of-fold as everything else it is compared against.

    The table backs off to a cell-only table exactly as `backoff_member` does,
    so an item whose hub says nothing degrades to the backoff it would have
    been rather than to noise.
    """
    alpha = float(CONFIG["hub_alpha"])
    depth = int(CONFIG["hub_depth"])
    hubs = hub_items(inst, fit_rows)
    if not hubs:
        return None
    index = {name: j for j, name in enumerate(inst.targets)}
    cap = int(CONFIG["backoff_rank_rows"])
    pick_rows = fit_rows
    if 0 < cap < len(pick_rows):
        pick_rows = pick_rows[np.random.default_rng(CONFIG["seed"] + 1).choice(
            len(pick_rows), cap, replace=False)]

    # One key per item over the columns its own ranking put first, built once.
    out = []
    for name in inst.targets:
        width = inst.width[name]
        y = inst.code[name][fit_rows]
        keep = y >= 0
        y_fit = y[keep]
        table = np.bincount(y_fit, minlength=width).astype(np.float64) \
            + CONFIG["smoothing"]
        crowd = table / table.sum()
        if y_fit.size < CONFIG["min_count"] * 4:
            out.append(np.tile(crowd, (len(rows), 1)))
            continue

        # the hub for this item: the candidate that says the most about it
        best, best_score = None, 0.0
        yr = np.maximum(inst.code[name][pick_rows], 0)
        for candidate in hubs:
            if candidate == name:
                continue
            hw = inst.width[candidate]
            hr = np.maximum(inst.code[candidate][pick_rows], 0)
            gain = mutual_information(hr, yr, hw, width) \
                - max(hw - 1, 0) * max(width - 1, 0) / (2.0 * len(pick_rows))
            if gain > best_score:
                best, best_score = candidate, gain
        if best is None or index[best] >= len(base) or base[index[best]] is None:
            out.append(np.tile(crowd, (len(rows), 1)))
            continue

        key_fit = np.zeros(y_fit.size, np.int64)
        key_out = np.zeros(len(rows), np.int64)
        size = 1
        for column in order[name][:depth]:
            card = inst.width[column]
            key_fit = key_fit * card + np.maximum(
                inst.code[column][fit_rows][keep], 0)
            key_out = key_out * card + np.maximum(inst.code[column][rows], 0)
            size *= card
        key_fit = np.minimum(key_fit, size - 1)
        key_out = np.minimum(key_out, size - 1)

        # cell-only table, which is what the hub table backs off to
        counts = np.bincount(key_fit * width + y_fit,
                             minlength=size * width).astype(np.float64)
        counts = counts.reshape(size, width)
        cell = (counts + alpha * crowd[None, :])
        cell /= cell.sum(axis=1, keepdims=True)

        hw = inst.width[best]
        hub_fit = np.maximum(inst.code[best][fit_rows][keep], 0)
        joint = np.bincount((hub_fit * size + key_fit) * width + y_fit,
                            minlength=hw * size * width).astype(np.float64)
        joint = joint.reshape(hw, size, width)
        joint += alpha * cell[None, :, :]
        joint /= joint.sum(axis=2, keepdims=True)

        share = base[index[best]]
        if share.shape[1] != hw:
            out.append(np.tile(crowd, (len(rows), 1)))
            continue
        vector = np.einsum("nh,hnk->nk", share, joint[:, key_out, :])
        out.append(vector / np.maximum(vector.sum(axis=1, keepdims=True), _EPS))
    return out


# ------------------------------------------------------------------ gates --

def gate_tables(inst, fit_rows):
    """How faithfully each declared gate actually predicts the sentinel.

    Two numbers per gated item, both measured on the visible rows, where the
    parent and the child are observed together:

        table[1] = P(child is the sentinel | the gate opened)
        table[0] = P(child is the sentinel | the gate did not open)

    A schema whose skip logic the data obeys gives (0, 1) and the hard rule
    back. UNHCR's does not: the root README records a median agreement of 0.83
    across items, with five multi-select groups whose sentinel appears only in
    round-2 rows and three branches of `s3_11` that overlap in the file. There
    the honest numbers are somewhere inside (0, 1), and measuring them is
    strictly better than asserting the corners -- which is what v2 did, and
    what cost it a cent of UNHCR.

    Fitted for GIVEN and PREDICT parents alike. The smoothing lands below the
    grader's floor at these row counts, so a gate that really is exact still
    reads as exact.
    """
    tables = {}
    for name in inst.targets:
        gate = inst.gate[name]
        if not gate:
            continue
        parent = gate["parent"]
        if parent not in inst.items:
            continue
        allowed = [i for i, value in enumerate(inst.cats[parent])
                   if value in set(gate["observed_if"])]
        if not allowed:
            continue
        open_fit = np.isin(inst.code[parent][fit_rows], allowed)
        sentinel = inst.code[name][fit_rows] == inst.width[name] - 1
        table = np.zeros(2)
        for side in (0, 1):
            rows = open_fit == bool(side)
            hits = float(sentinel[rows].sum()) + CONFIG["gate_smoothing"]
            table[side] = hits / (rows.sum() + 2 * CONFIG["gate_smoothing"])
        tables[name] = (allowed, table)
    return tables


def gate_models(inst, fit_rows):
    """The same question as `gate_tables`, asked of each kind of respondent.

    v2 measures a gate's fidelity as two scalars -- P(sentinel | opened) and
    P(sentinel | not opened) -- pooled over the whole sample. The UNHCR file
    is documented to break exactly that assumption: in five multi-select
    families (6.4, 6.6, 6.11, 7A.7, 7B.1, 59 of the 213 scored items) round 1
    wrote a 0 for the respondents its routing skipped and round 2 marked them
    skipped, so

        P(sentinel | not opened, wave = 1) = 0
        P(sentinel | not opened, wave = 2) = 1

    and the single pooled number lands near the round mix, roughly 0.4, which
    is wrong for every row of both waves. `wave` is a GIVEN column, so the
    right number is knowable per row; nothing in the schema's one-parent gate
    format can say so, which is why it has to be measured.

    So the two scalars become two functions of the GIVEN block: a Dirichlet
    backoff from the pooled rate, down the side of the gate first and then the
    `gate_cond_depth` GIVEN columns carrying the most mutual information with
    the sentinel indicator of this item. Ranking against the sentinel rather
    than against the answer is the point -- `wave` says little about what a
    household replied and everything about whether the cell was marked.

    An item whose sentinel really is a pure function of the gate backs off to
    the pooled pair and reproduces `gate_tables`, so this can only add
    resolution where the data carries some.
    """
    alpha = CONFIG["gate_cond_alpha"]
    depth = max(0, int(CONFIG["gate_cond_depth"]))
    smooth = CONFIG["gate_smoothing"]
    models = {}
    for name in inst.targets:
        gate = inst.gate[name]
        if not gate:
            continue
        parent = gate["parent"]
        if parent not in inst.items:
            continue
        allowed = [i for i, value in enumerate(inst.cats[parent])
                   if value in set(gate["observed_if"])]
        if not allowed:
            continue
        side = np.isin(inst.code[parent][fit_rows], allowed).astype(np.int64)
        sentinel = (inst.code[name][fit_rows]
                    == inst.width[name] - 1).astype(np.float64)
        mark = sentinel.astype(np.int64)
        if CONFIG["gate_rank"] == "conditional":
            # Same correction as `rank_given`, for the same reason, on the
            # table that carries this instrument's documented defect. The
            # chain here starts from the gate's own side, so the first column
            # is already being chosen conditionally on it -- which is the
            # point: `wave` is worth nothing about *whether the gate opened*
            # and everything about how a closed gate was written down.
            chosen, rest = [], list(inst.given)
            key, kc = side.copy(), 2
            for _ in range(min(depth, len(rest))):
                best, score = None, -np.inf
                for g in rest:
                    value = conditional_mi(
                        key, kc, np.maximum(inst.code[g][fit_rows], 0),
                        inst.width[g], mark, 2,
                        bool(CONFIG["mi_bias_correct"]))
                    if value > score:
                        best, score = g, value
                chosen.append(best)
                rest.remove(best)
                key = key * inst.width[best] + np.maximum(
                    inst.code[best][fit_rows], 0)
                key, inverse = np.unique(key, return_inverse=True)
                key, kc = inverse.astype(np.int64), key.size
            columns = chosen
        else:
            columns = sorted(
                inst.given, reverse=True,
                key=lambda g: mutual_information(inst.code[g][fit_rows],
                                                 mark, inst.width[g], 2)
                - (bool(CONFIG["mi_bias_correct"])
                   * max(inst.width[g] - 1, 0)
                   / (2.0 * max(len(fit_rows), 1))))[:depth]

        base = (sentinel.sum() + smooth) / (sentinel.size + 2 * smooth)
        estimate = np.full(1, base)
        key = np.zeros(sentinel.size, np.int64)
        for codes, card in [(side, 2)] + [(np.maximum(
                inst.code[column][fit_rows], 0), inst.width[column])
                for column in columns]:
            key = key * card + codes
            size = estimate.size * card
            hits = np.bincount(key, weights=sentinel, minlength=size)
            total = np.bincount(key, minlength=size).astype(np.float64)
            parent_estimate = np.repeat(estimate, card)
            estimate = ((hits + alpha * parent_estimate)
                        / (total + alpha))
        models[name] = (allowed, estimate.reshape(2, -1), columns)
    return models


def gate_groups(inst, fit_rows):
    """`gate_models` again, but fitted on everyone the same routing skipped.

    A gate is not a property of an item, it is a property of a question the
    interviewer either put or did not put. Every item hanging off the same
    parent value is marked -- or not marked -- by one event, and UNHCR's five
    multi-select families are 59 items reading out five events. Fitting 59
    separate tables from 8,279 rows each, when what is being estimated is the
    same probability 59 times over, throws away the only thing that would pay
    for a third and fourth stratifying column: rows.

    So the routing event is fitted once per group, `gate_group_depth` columns
    deep, on the group's rows stacked, and each item takes that table shifted
    on the logit by however far its own rate sits from the group's. Items whose
    sentinel really is their own -- the documented overlap between the three
    `s3_11` branches is exactly that -- keep the offset and lose nothing, and
    the whole family is on the roster next to the per-item version so the
    stack can prefer either, per item.

    Returns the same (allowed, table, columns) shape `gate_models` returns, so
    `gate_cond_share` and `gate_refine` read it without knowing the difference.
    """
    alpha = CONFIG["gate_cond_alpha"]
    depth = max(0, int(CONFIG["gate_group_depth"]))
    smooth = CONFIG["gate_smoothing"]

    models = {}
    for (parent, allowed), members in gate_families(inst).items():
        side = np.isin(inst.code[parent][fit_rows],
                       list(allowed)).astype(np.int64)
        flags = {name: (inst.code[name][fit_rows]
                        == inst.width[name] - 1).astype(np.float64)
                 for name in members}
        pooled = np.concatenate([flags[name] for name in members])
        stacked = np.concatenate([side] * len(members))
        columns = sorted(
            inst.given, reverse=True,
            key=lambda g: mutual_information(
                np.concatenate([inst.code[g][fit_rows]] * len(members)),
                pooled.astype(np.int64), inst.width[g], 2))[:depth]

        base = (pooled.sum() + smooth) / (pooled.size + 2 * smooth)
        estimate = np.full(1, base)
        key = np.zeros(pooled.size, np.int64)
        for codes, card in [(stacked, 2)] + [(np.concatenate(
                [np.maximum(inst.code[column][fit_rows], 0)] * len(members)),
                inst.width[column]) for column in columns]:
            key = key * card + codes
            size = estimate.size * card
            hits = np.bincount(key, weights=pooled, minlength=size)
            total = np.bincount(key, minlength=size).astype(np.float64)
            estimate = ((hits + alpha * np.repeat(estimate, card))
                        / (total + alpha))
        table = np.clip(estimate.reshape(2, -1), 1e-6, 1 - 1e-6)
        group_logit = np.log(base / (1.0 - base))

        for name in members:
            own = (flags[name].sum() + smooth) / (flags[name].size + 2 * smooth)
            own = min(max(own, 1e-6), 1 - 1e-6)
            offset = np.log(own / (1.0 - own)) - group_logit
            moved = 1.0 / (1.0 + np.exp(
                -(offset + np.log(table / (1.0 - table)))))
            models[name] = (list(allowed), np.clip(moved, 1e-6, 1 - 1e-6),
                            columns)
    return models


def materialise(inst, models, rows):
    """Turn (allowed, table, columns) into (allowed, closed, opened) per row.

    Every conditional gate family ends up as two numbers per row -- P(sentinel)
    if the gate did not open and P(sentinel) if it did -- and `gate_refine`
    wants nothing else. Reading the backoff tables out here means a family that
    is not a table at all can join the roster behind the same interface, which
    is what `gate_trees` does.
    """
    out = {}
    for name, (allowed, table, columns) in models.items():
        index = np.zeros(len(rows), np.int64)
        for column in columns:
            index = index * inst.width[column] + np.maximum(
                inst.code[column][rows], 0)
        index = np.minimum(index, table.shape[1] - 1)
        out[name] = (allowed, table[0, index], table[1, index])
    return out


def pos_members(inst, fit_rows, rows, windows):
    """The visible rows nearest this one in file order, voting.

    A household survey is collected in clusters -- one interviewer, one day,
    one camp -- and a file written in fieldwork order puts those neighbours
    next to each other. ERPIS was fielded over 67 interview dates and neither
    the date nor the interviewer is in the GIVEN block, so if the delivery
    preserves that order it is the one covariate on the frame that the schema
    does not already carry.

    A moving window over the visible rows sorted by position, backed off to the
    crowd exactly like the other counting members. Two widths, because the size
    of a cluster is not known. If the roles were assigned in a way that
    destroys the order, or the order was never meaningful, these members carry
    the crowd marginal and the stack drops them item by item.
    """
    alpha = CONFIG["pos_alpha"]
    order = np.argsort(inst.position[fit_rows], kind="stable")
    ranks = np.searchsorted(inst.position[fit_rows][order],
                            inst.position[rows])
    out = []
    for name in inst.targets:
        width = inst.width[name]
        y = inst.code[name][fit_rows][order]
        seen = y >= 0
        table = np.bincount(y[seen], minlength=width).astype(np.float64)
        crowd = (table + CONFIG["smoothing"])
        crowd /= crowd.sum()
        cumulative = np.zeros((y.size + 1, width))
        hits = np.zeros((y.size, width))
        hits[np.flatnonzero(seen), y[seen]] = 1.0
        cumulative[1:] = np.cumsum(hits, axis=0)
        members = []
        for window in windows:
            half = max(int(window) // 2, 1)
            low = np.clip(ranks - half, 0, y.size)
            high = np.clip(ranks + half, 0, y.size)
            counts = cumulative[high] - cumulative[low]
            estimate = (counts + alpha * crowd[None, :])
            members.append(estimate / estimate.sum(axis=1, keepdims=True))
        out.append(members)
    return [[out[j][i] for j in range(len(inst.targets))]
            for i in range(len(windows))]


def gate_families(inst):
    """Gated items grouped by the routing event that marks them."""
    families = {}
    for name in inst.targets:
        gate = inst.gate[name]
        if not gate or gate["parent"] not in inst.items:
            continue
        allowed = tuple(i for i, value in enumerate(inst.cats[gate["parent"]])
                        if value in set(gate["observed_if"]))
        if allowed:
            families.setdefault((gate["parent"], allowed), []).append(name)
    return families


def gate_trees(inst, fit_rows, rows):
    """The routing event as a boosted tree, one per group.

    `gate_groups` buys depth by pooling the group's rows, but it still walks a
    fixed chain of columns chosen one at a time by mutual information. The
    answer half of the roster has just been told, by the out-of-fold
    likelihood, that a chain is not how this instrument's conditionals are
    shaped. There is no reason the routing half is different, and it is the
    half that carries most of the score: an UNHCR cell's information is mostly
    the one number that says whether the cell is blank.

    Fitted as a regression on the group's **sentinel share per respondent** --
    the fraction of the group's items that came back marked -- rather than on
    the group's rows stacked item by item. Same conditional expectation, one
    row per respondent instead of 59, and the 27 groups cost a couple of
    seconds between them.

    `side` is a feature rather than a split of the data, so one fit answers
    both branches: predict with it held at 0 and at 1.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    matrix, categorical = given_codes(inst)
    params = dict(CONFIG["gate_tree_params"])
    out = {}
    for (parent, allowed), members in gate_families(inst).items():
        side = np.isin(inst.code[parent][fit_rows],
                       list(allowed)).astype(np.float64)
        flags = {name: (inst.code[name][fit_rows]
                        == inst.width[name] - 1).astype(np.float64)
                 for name in members}
        target = np.mean([flags[name] for name in members], axis=0)
        base = min(max(float(target.mean()), 1e-6), 1 - 1e-6)
        train = np.column_stack([matrix[fit_rows], side])
        read = matrix[rows]
        try:
            model = HistGradientBoostingRegressor(
                categorical_features=categorical, early_stopping=False,
                random_state=CONFIG["seed"], **params)
            model.fit(train, target)
            group = [np.clip(model.predict(np.column_stack(
                [read, np.full(len(rows), float(s))])), 1e-4, 1 - 1e-4)
                for s in (0, 1)]
        except Exception:
            group = [np.full(len(rows), base), np.full(len(rows), base)]
        group_logit = [np.log(g / (1.0 - g)) for g in group]
        shift = np.log(base / (1.0 - base))
        for name in members:
            own = float((flags[name].sum() + CONFIG["gate_smoothing"])
                        / (flags[name].size + 2 * CONFIG["gate_smoothing"]))
            own = min(max(own, 1e-6), 1 - 1e-6)
            offset = np.log(own / (1.0 - own)) - shift
            moved = [np.clip(1.0 / (1.0 + np.exp(-(offset + logit))),
                             1e-6, 1 - 1e-6) for logit in group_logit]
            out[name] = (list(allowed), moved[0], moved[1])
    return out


def gate_cond_share(inst, name, rows, models):
    """P(sentinel | gate closed, x) and P(sentinel | gate open, x) for `rows`."""
    return models[name][1], models[name][2]


def gate_refine(inst, member, rows, tables, conditional=False):
    """Rebuild every gated item around the skip logic, parents before children.

    Whether the gate opened is known or guessed, depending on the parent:

      parent GIVEN    everybody's parent answer is visible, so this is an
                      indicator, not a guess.
      parent PREDICT  the parent is hidden too, so its own predicted
                      distribution supplies the probability, summed over the
                      values that open the gate.

    Either way the opening is then run through the item's measured fidelity
    rather than assumed to determine the sentinel:

        P(sentinel) = P(opened) * table[1] + (1 - P(opened)) * table[0]

    With an exact gate table[1] is ~0 and table[0] is ~1, which is the hard
    rule `1 - P(opened)` -- so this only ever differs where the data says the
    declared gate is not what happened. `CONFIG["soft_gate"]` off restores the
    assertion, for ablation.

    Walking in depth order and reading the parent's **refined** distribution is
    what makes a chain compound: UNHCR has 23 items two or more gates deep, and
    reading the unrefined parent silently truncates the chain at one level.

    `conditional` reads `tables` as the per-row models `gate_models` fits
    rather than the two scalars `gate_tables` measures. Everything else about
    the walk is the same, which is what makes the two versions a clean pair on
    the roster for the stack to choose between, per item.

    Returns the rebuilt member and the P(sentinel) it used, which the hurdle
    members reuse rather than recomputing.
    """
    index = {name: j for j, name in enumerate(inst.targets)}
    out = list(member)
    sentinel = {}
    for name in sorted(inst.targets, key=lambda n: inst.depth[n]):
        gate = inst.gate[name]
        if not gate or name not in tables:
            continue
        j = index[name]
        parent = gate["parent"]
        if conditional:
            allowed = tables[name][0]
            closed, opened_rate = gate_cond_share(inst, name, rows, tables)
        else:
            allowed, table = tables[name]
        if parent in inst.given:
            opened = np.isin(inst.code[parent][rows], allowed).astype(float)
        elif parent in index:
            opened = out[index[parent]][:, allowed].sum(axis=1)
        else:
            continue
        if conditional:
            share = opened * opened_rate + (1.0 - opened) * closed
        elif CONFIG["soft_gate"]:
            share = opened * table[1] + (1.0 - opened) * table[0]
        else:
            share = np.where(opened > 0.5, table[1], table[0]) \
                if parent in inst.given else 1.0 - opened
        share = np.clip(share, 1e-6, 1 - 1e-6).astype(np.float32)
        sentinel[name] = share
        body = out[j][:, :-1]
        body = body / np.maximum(body.sum(axis=1, keepdims=True), _EPS)
        out[j] = np.concatenate(
            [body * (1.0 - share)[:, None], share[:, None]], axis=1)
    return out, sentinel


def hurdle_member(inst, answer, sentinel, fallback):
    """P(answer | asked) from the asked rows, times P(asked) from the gate.

    The K+1-way softmax that the other members fit has to spend its capacity,
    and its regularisation, on the sentinel as well as on the answer. When a
    gated item is a sentinel for most respondents -- and under `s3_11` most of
    the 73 branch items are, for two thirds of the sample -- what is left to
    learn "what they said when asked" is a small fraction of the rows, shrunk
    as if it were all of them. Fitting the answer on the asked rows alone and
    multiplying by the gate's P(asked) puts each half on its own footing.
    """
    out = []
    for j, name in enumerate(inst.targets):
        share = sentinel.get(name)
        if not inst.gate[name]:
            out.append(answer[j])
            continue
        if share is None:
            # gate_refine could not place this item -- an option list that no
            # longer contains what observed_if names, say. The answer half is
            # a slot short without its sentinel, and a wrong width fails the
            # whole submission, so take the ordinary refined member instead.
            out.append(fallback[j])
            continue
        body = answer[j]
        body = body / np.maximum(body.sum(axis=1, keepdims=True), _EPS)
        out.append(np.concatenate(
            [body * (1.0 - share)[:, None], share[:, None]], axis=1))
    return out


def shift_weights(inst):
    """Raking weights on the visible rows, matching the held-out GIVEN mix.

    Returns an array over `inst.fit_rows` with mean 1, or None when the two
    blocks are indistinguishable on every GIVEN column -- the common case,
    and then nothing downstream changes at all.
    """
    if not CONFIG.get("shift_weights"):
        return None
    fit, out = inst.fit_rows, inst.out_rows
    if len(out) < 50 or len(fit) < 200 or not inst.given:
        return None
    try:
        from scipy.stats import chi2
    except Exception:
        return None
    alpha = float(CONFIG["shift_alpha"]) / max(len(inst.given), 1)
    coded = {}
    shifted = []
    for name in inst.given:
        width = inst.width[name] + 1                   # unseen has a level
        cf = inst.code[name][fit]
        co = inst.code[name][out]
        cf = np.where(cf < 0, width - 1, cf)
        co = np.where(co < 0, width - 1, co)
        coded[name] = (cf, co, width)
        tf = np.bincount(cf, minlength=width).astype(np.float64)
        to = np.bincount(co, minlength=width).astype(np.float64)
        tot = tf + to
        keep = tot > 0
        tf, to, tot = tf[keep], to[keep], tot[keep]
        nf, no = tf.sum(), to.sum()
        ef = tot * nf / (nf + no)
        eo = tot * no / (nf + no)
        small = eo < 5.0
        if small.any():
            tf = np.append(tf[~small], tf[small].sum())
            to = np.append(to[~small], to[small].sum())
            ef = np.append(ef[~small], ef[small].sum())
            eo = np.append(eo[~small], eo[small].sum())
        if tf.size < 2:
            continue
        stat = float(((tf - ef) ** 2 / np.maximum(ef, 1e-9)).sum()
                     + ((to - eo) ** 2 / np.maximum(eo, 1e-9)).sum())
        p = float(chi2.sf(stat, tf.size - 1))
        if p < alpha:
            shifted.append(name)
    _STAGE["shifted"] = list(shifted)
    if not shifted:
        return None
    lo, hi = CONFIG["shift_clip"]
    w = np.ones(len(fit))
    for _ in range(int(CONFIG["shift_iterations"])):
        for name in shifted:
            cf, co, width = coded[name]
            target = (np.bincount(co, minlength=width) + 0.5) \
                / (len(co) + 0.5 * width)
            current = np.bincount(cf, weights=w, minlength=width) / w.sum()
            factor = np.where(current > 0, target / np.maximum(current, 1e-12),
                              1.0)
            w = np.clip(w * factor[cf], lo, hi)
        w = w / w.mean()
    _STAGE["shift_ess"] = float(w.sum() ** 2 / (w ** 2).sum())
    return w


def em_weights(likelihood, n_members, n_items, shared=None, row_weight=None):
    """Per item, the simplex weights that maximise the out-of-fold likelihood.

    Plain EM for a mixture of fixed components: the responsibility a member
    takes for a cell becomes its next weight. A weak Dirichlet prior keeps a
    member that never wins from going exactly to zero, which matters because
    the weights are fitted on models trained on a fraction of the rows and
    applied to models trained on all of them.

    `shared` is the same fit run once over every item at once, and it enters
    each item's prior worth `em_pool` cells. Without it an item's hundred
    weights are estimated from that item's cells alone, which is thin enough
    that adding defensible members can lose score -- measured, not feared.
    """
    prior = CONFIG["em_prior"]
    pool = float(CONFIG["em_pool"]) if shared is not None else 0.0
    base = (prior + pool * np.asarray(shared).reshape(-1, 1, 1)
            if shared is not None else prior)
    iterations = CONFIG["em_iterations"]
    if torch_available():
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            lik = torch.as_tensor(np.ascontiguousarray(likelihood),
                                  dtype=torch.float32, device=device
                                  ).clamp_min(1e-12)
            w = torch.full((n_members, 1, n_items), 1.0 / n_members,
                           device=device)
            rw = None
            if row_weight is not None:
                rw = torch.as_tensor(np.ascontiguousarray(row_weight),
                                     dtype=torch.float32, device=device
                                     ).reshape(1, -1, 1)
            for _ in range(iterations):
                post = lik * w
                post = post / post.sum(dim=0, keepdim=True).clamp_min(_EPS)
                if rw is not None:
                    post = post * rw
                counts = post.sum(dim=1, keepdim=True) + (
                    torch.as_tensor(base, dtype=torch.float32, device=device)
                    if shared is not None else prior)
                new = counts / counts.sum(dim=0, keepdim=True)
                shift = (new - w).abs().max().item()
                w = new
                if shift < 1e-6:
                    break
            out = w[:, 0, :].t().cpu().numpy().astype(np.float64)
            del lik, w
            if device == "cuda":
                torch.cuda.empty_cache()
            return out
        except Exception:
            pass

    w = np.full((n_items, n_members), 1.0 / n_members)
    lik = np.maximum(likelihood, 1e-12)                 # [M, N, J]
    for _ in range(iterations):
        post = lik * w.T[:, None, :]
        post /= np.maximum(post.sum(axis=0), _EPS)
        if row_weight is not None:
            post *= np.asarray(row_weight, np.float32)[None, :, None]
        counts = post.sum(axis=1).T + (
            prior + pool * shared[None, :] if shared is not None else prior)
        new = counts / counts.sum(axis=1, keepdims=True)
        if np.abs(new - w).max() < 1e-6:
            return new
        w = new
    return w


_TORCH = None


def torch_available():
    global _TORCH
    if _TORCH is None:
        try:
            import torch                                          # noqa: F401
            _TORCH = True
        except Exception:
            _TORCH = False
    return _TORCH


def sklearn_available():
    try:
        import sklearn                                       # noqa: F401
        return bool(CONFIG["gbm"])
    except Exception:
        return False


def gbm_floor_list():
    return list(CONFIG.get("gbm_floors") or [CONFIG["gbm_floor"]])


def gbm_name(index, slot):
    """gbm0 is the shipped floor; the extra floors get their own names."""
    floors = gbm_floor_list()
    return "gbm%d" % index if slot == 0 else "gbm%df%g" % (index, floors[slot])


def gbm_names():
    if not sklearn_available():
        return []
    return [gbm_name(i, k) for i in range(len(CONFIG["gbm"]))
            for k in range(len(gbm_floor_list()))]


def backoff_name(depth, slot):
    alphas = list(CONFIG.get("backoff_alphas") or [CONFIG["backoff_alpha"]])
    return "backoff%d" % depth if slot == 0 \
        else "backoff%d@%s" % (depth, alphas[slot])


def fitted_names():
    """The members a hurdle can be built on: they can be fitted answer-only."""
    names = []
    if torch_available():
        names = ["logistic@%g" % r for r in CONFIG["ridge_grid"]]
        for width in CONFIG["trunk_widths"]:
            names += ["trunk%d@%g" % (width, r) for r in CONFIG["trunk_ridges"]]
        for width in CONFIG["deep_widths"]:
            names += ["deep%d@%g" % (width, r) for r in CONFIG["deep_ridges"]]
    if CONFIG["gbm_hurdle"]:
        names += ["gbm%d" % i for i in range(len(CONFIG["gbm"]))] \
            if sklearn_available() else []
    if CONFIG["tabpfn"] and CONFIG.get("tabpfn_hurdle"):
        names += ["tabpfn"]
    return names


def roster():
    """The estimator names, in order, decided before the run.

    Every fold and the final refit return exactly this list, so the stack's
    weights always line up with the members they were fitted for. A member that
    fails or runs out of time is filled with the crowd marginal rather than
    dropped, which shifts weight onto the other members instead of silently
    renumbering them.
    """
    alphas = list(CONFIG.get("backoff_alphas") or [CONFIG["backoff_alpha"]])
    names = ["marginal"] + [backoff_name(d, k)
                            for k in range(len(alphas))
                            for d in CONFIG["backoff_depths"]]
    names += ["logadd%d" % i for i in range(len(CONFIG["logadd"]))]
    names += ["rgm%dd%d" % (c, d) for c in CONFIG["regime_classes"]
              for d in CONFIG["regime_depths"]]
    if torch_available():
        names += ["knn%d" % k for k in CONFIG["knn_k"]]
    fitted = fitted_names()
    names += [n for n in fitted if n not in names]
    names += [n for n in gbm_names() if n not in names]
    if CONFIG["hub"]:
        names += ["hub"]
    if CONFIG["tabpfn"] and "tabpfn" not in names:
        names += ["tabpfn"]
    if CONFIG["pos_members"]:
        names += ["pos%d" % w for w in CONFIG["pos_windows"]]
    if torch_available():
        names += ["latent%d" % c for c in CONFIG["latent_classes"]]
        names += ["latent%dJ" % c for c in CONFIG["latent_joint_classes"]]
    # Sharpening only the fitted families: a linear pool is never sharper than
    # its members. Off by default -- it measured as no better either way, and
    # the fitted exponent in `calibrate` is the same idea done once, on the
    # pool, against held-out cells.
    names += ["%s^%g" % (n, g) for n in names if "@" in n or n.startswith("latent")
              for g in CONFIG["sharpen"]]
    gated = [n + "+gate" for n in names]
    # The conditional gate is a second opinion, not a replacement: both go on
    # the roster and the stack decides per item which reading of the skip logic
    # the data supports.
    if CONFIG["gate_cond"]:
        gated += [n + "+gateC" for n in names]
    extra = [n for n in names
             if n.startswith(tuple(CONFIG["gate_extra_bases"]))]
    if CONFIG["gate_group"]:
        gated += [n + "+gateG" for n in extra]
    if CONFIG["gate_tree"] and sklearn_available():
        gated += [n + "+gateT" for n in extra]
    hurdle = ["hurdle:" + n for n in fitted] if CONFIG["hurdle"] else []
    if CONFIG["hurdle"] and CONFIG["gate_cond"]:
        hurdle += ["hurdleC:" + n for n in fitted]
    if CONFIG["hurdle"] and CONFIG["gate_group"]:
        hurdle += ["hurdleG:" + n for n in fitted]
    if CONFIG["hurdle"] and CONFIG["gate_tree"] and sklearn_available():
        hurdle += ["hurdleT:" + n for n in fitted]
    return names + gated + hurdle


def sharpen(member, gamma):
    """The same member, made more (or less) decisive. p^gamma, renormalised."""
    out = []
    for p in member:
        q = np.power(np.maximum(p, _EPS), gamma)
        out.append(q / np.maximum(q.sum(axis=1, keepdims=True), _EPS))
    return out


def _base_name(name):
    """A roster name with its gate suffix and hurdle prefix stripped."""
    for prefix in ("hurdleT:", "hurdleG:", "hurdleC:", "hurdle:"):
        if name.startswith(prefix):
            return name[len(prefix):]
    for suffix in ("+gateT", "+gateG", "+gateC", "+gate"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


# Every arm the screen decides: which roster names it owns, and what enabling
# it does to CONFIG. Owning a name is what lets one build answer eleven
# questions -- an arm is removed from the mixture at the likelihood matrix,
# not rebuilt.
def c_widths():
    return CONFIG["trunk_wide_widths"]


ARMS = {
    "knn_small":    (lambda n: _base_name(n) in ("knn1", "knn5", "knn15"),
                     lambda c: {"knn_k": sorted(set(c["knn_small_k"])
                                                | set(c["knn_k"]))}),
    "pos":          (lambda n: _base_name(n) in ("pos120", "pos800"),
                     lambda c: {"pos_members": True}),
    "gateG":        (lambda n: n.endswith("+gateG") or n.startswith("hurdleG:"),
                     lambda c: {"gate_group": True}),
    "gateT":        (lambda n: n.endswith("+gateT") or n.startswith("hurdleT:"),
                     lambda c: {"gate_tree": {"on": True}}),
    "hub":          (lambda n: _base_name(n) == "hub",
                     lambda c: {"hub": True}),
    "tabpfn":       (lambda n: _base_name(n) == "tabpfn",
                     lambda c: {"tabpfn": True}),
    "sharp_tables": (lambda n: "@" in _base_name(n)
                     and _base_name(n).startswith("backoff"),
                     lambda c: {"backoff_alphas": c["backoff_sharp_alphas"]}),
    "sharp_trees":  (lambda n: "f" in _base_name(n).split("gbm")[-1]
                     if _base_name(n).startswith("gbm") else False,
                     lambda c: {"gbm_floors": c["gbm_sharp_floors"]}),
    "backoff_deep": (lambda n: _base_name(n) in ("backoff6", "backoff7"),
                     lambda c: {"backoff_depths":
                                sorted(set(c["backoff_depths"])
                                       | set(c["backoff_deep_depths"]))}),
    "latent_joint": (lambda n: _base_name(n).startswith("latent")
                     and _base_name(n).endswith("J"),
                     lambda c: {"latent_joint_classes": c["latent_joint_grid"]}),
    "deep":         (lambda n: _base_name(n).startswith("deep"),
                     lambda c: {"deep_widths": c["deep_grid"]}),
    "trunk_wide":   (lambda n: any(_base_name(n).startswith("trunk%d@" % w)
                                   for w in c_widths()),
                     lambda c: {"trunk_widths":
                                sorted(set(c["trunk_widths"])
                                       | set(c["trunk_wide_widths"]))}),
    "gbm_decorr":   (lambda n: _base_name(n).startswith("gbm1"),
                     lambda c: {"gbm": c["gbm"] + [c["gbm_grid"][3]]}),
    "latent64":     (lambda n: _base_name(n) == "latent64",
                     lambda c: {"latent_classes":
                                sorted(set(c["latent_classes"])
                                       | set(c["latent64_classes"]))}),
    "logadd":       (lambda n: _base_name(n).startswith("logadd"),
                     lambda c: {"logadd": c["logadd_grid"]}),
    "regime":       (lambda n: _base_name(n).startswith("rgm"),
                     lambda c: {"regime_classes": c["regime_grid"]}),
}
def fit_roster_to_memory(inst, n_readout):
    """Shed arms, widest first, until the roster fits the memory budget."""
    slots = sum(inst.width[name] for name in inst.targets)
    def estimate():
        members = len(roster())
        member_bytes = members * n_readout * slots * 4
        likelihood_bytes = members * len(inst.fit_rows) * len(inst.targets) * 4
        bag_bytes = members * len(inst.out_rows) * slots * 4
        return (member_bytes + likelihood_bytes + bag_bytes) / 1e9
    shed = []
    for arm in CONFIG["shed_order"]:
        if estimate() <= CONFIG["member_budget_gb"]:
            break
        patch = ARMS[arm][1]
        if patch is None:
            continue
        for key, value in {"gateT": {"gate_tree": {}},
                           "gateG": {"gate_group": False},
                           "latent64": {"latent_classes":
                                        [c for c in CONFIG["latent_classes"]
                                         if c not in CONFIG["latent64_classes"]]},
                           "logadd": {"logadd": []},
                           "backoff_deep": {"backoff_depths":
                                            [d for d in CONFIG["backoff_depths"]
                                             if d not in
                                             CONFIG["backoff_deep_depths"]]},
                           "knn_small": {"knn_k":
                                         [k for k in CONFIG["knn_k"]
                                          if k not in CONFIG["knn_small_k"]]},
                           "tabpfn": {"tabpfn": False},
                           "deep": {"deep_widths": []},
                           "trunk_wide": {"trunk_widths":
                                          [w for w in CONFIG["trunk_widths"]
                                           if w not in
                                           CONFIG["trunk_wide_widths"]]},
                           "gbm_decorr": {"gbm": CONFIG["gbm"][:1]},
                           "latent_joint": {"latent_joint_classes": []},
                           "regime": {"regime_classes": []},
                           }[arm].items():
            CONFIG[key] = value
        shed.append(arm)
    return shed, estimate()


def prune_arms(names, cells, n_ship, seed):
    """Which optional families earned their width, asked of the delivered rows.

    The one thing this repo has never had is a ruler in the regime the real
    instrument is in. Every fixture it owns caps out at a ceiling of 0.79
    where the board's leader sits at 0.92, and `fit_cell` reaches even that by
    drawing an independent Dirichlet per GIVEN cell -- a target with nothing
    to generalise across cells, which is the opposite of a questionnaire.
    `headroom.py` on it returns a shortfall that is flat across all 213 items,
    which is the signature of that generator and not of anything a submission
    could fix. A family ranking measured there does not transfer. The
    alternative the team had fallen back on -- one leaderboard number a day --
    is slower by three orders of magnitude, and it is what the competition
    rules tell participants not to lean on ("exploit repeated score feedback").

    So the ranking is measured here, on the rows the container was handed.
    `cells` is the out-of-fold likelihood the members already produced for the
    weights, so this costs no fitting at all: an arm is deleted from the
    mixture, the weights are refitted without it, and the two mixtures are
    compared on cells that neither fit saw. Cross-fitted across halves of the
    out-of-fold rows, because a wider roster flatters itself on the cells its
    own weights were chosen on -- which is exactly the failure that made
    `gate_group` and `gate_tree` measure worse when they were added on their
    merits.

    The comparison is paired per respondent and the statistic is the
    cluster-robust t of that pairing, which is the right unit: one respondent
    contributes 213 cells and they move together.
    """
    n_members, n_rows, n_items = cells.shape
    dropped = np.zeros(n_members, bool)
    if not CONFIG["screen"] or n_rows < CONFIG["screen_min_rows"]:
        return dropped, {}

    rng = np.random.default_rng(seed + 7)
    order = rng.permutation(n_rows)
    cut = n_rows // 2
    halves = ((order[:cut], order[cut:]), (order[cut:], order[:cut]))

    # The estimator under test has to be the estimator that ships. The mixture
    # this instrument sends is a per-item one fitted on all `n_ship`
    # out-of-fold rows, and a per-item mixture on half a fold's rows is a much
    # more overfitted object -- it would charge every family six times the
    # width it actually costs, and delete all of them. What decides how far a
    # simplex fit overfits is the prior's weight against the rows, so the
    # prior is scaled by exactly the ratio of rows and the two fits shrink the
    # same amount. That, and matching the iteration count, is what makes the
    # verdict here a verdict about the shipped model.
    scale = max(float(n_ship) / max(len(halves[0][0]), 1), 1.0)
    prior = float(CONFIG["em_prior"]) * scale
    iterations = int(CONFIG["em_iterations"])

    # The whole test is one tensor and a few dozen EM fits on it, so it lives
    # on the GPU for its duration: sliced on the host it is half a gigabyte of
    # memcpy per arm, and there are twenty arms to try.
    torch = None
    if torch_available():
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                torch = _torch
        except Exception:
            torch = None

    if torch is not None:
        device = "cuda"
        panels = [torch.as_tensor(np.ascontiguousarray(cells[:, rows, :]),
                                  device=device).clamp_min(1e-12)
                  for rows in (halves[0][0], halves[0][1])]

        def evaluate(mask):
            index = np.flatnonzero(mask)
            if index.size < 2:
                return None
            pick = torch.as_tensor(index, device=device)
            total = np.zeros(n_rows)
            for half, (fit_rows, read_rows) in enumerate(halves):
                lik = panels[half][pick]
                w = torch.full((index.size, 1, n_items), 1.0 / index.size,
                               device=device)
                for step in range(iterations):
                    post = lik * w
                    post = post / post.sum(0, keepdim=True).clamp_min(1e-30)
                    counts = post.sum(1, keepdim=True) + prior
                    new = counts / counts.sum(0, keepdim=True)
                    # Reading the convergence test costs a device sync, and
                    # there are twenty arms times two halves of these loops.
                    done_now = step % 25 == 24 and \
                        (new - w).abs().max().item() < 1e-6
                    w = new
                    if done_now:
                        break
                pooled = (panels[1 - half][pick] * w).sum(0).clamp_min(1e-12)
                total[read_rows] = pooled.log().sum(1).double().cpu().numpy()
            return total
    else:
        saved_prior = CONFIG["em_prior"]

        def evaluate(mask):
            index = np.flatnonzero(mask)
            if index.size < 2:
                return None
            block = cells[index]
            total = np.zeros(n_rows)
            CONFIG["em_prior"] = prior
            try:
                for fit_rows, read_rows in halves:
                    weight = em_weights(block[:, fit_rows, :], index.size,
                                        n_items)
                    got = np.einsum("jm,mnj->nj", weight,
                                    block[:, read_rows, :])
                    total[read_rows] = np.log(np.maximum(got, _EPS)).sum(axis=1)
            finally:
                CONFIG["em_prior"] = saved_prior
            return total

    everything = evaluate(np.ones(n_members, bool))
    if everything is None:
        return dropped, {}

    def paired(other):
        gap = everything - other
        error = gap.std(ddof=1) / max(np.sqrt(gap.size), 1.0)
        return float(gap.mean() / error) if error > 0 else 0.0

    verdict, owned, order = {}, {}, []
    for arm in CONFIG["screen_arms"]:
        mine = np.array([ARMS[arm][0](name) for name in names])
        if not mine.any():
            continue
        without = evaluate(~mine)
        if without is None:
            verdict[arm] = (float("nan"), int(mine.sum()))
            continue
        t = paired(without)
        verdict[arm], owned[arm] = (t, int(mine.sum())), mine
        order.append((t, arm))

    # Leave-one-out cannot be read arm by arm when two arms answer the same
    # need: `sharp_tables` and `sharp_trees` both make a near-deterministic
    # cell sharp, so each can measure worthless while the other is present and
    # both get deleted, taking the effect with them. So the deletions are
    # applied cumulatively, worst first, and each one has to leave the mixture
    # no worse than the whole roster was -- an arm whose removal starts
    # costing something is put back.
    threshold = float(CONFIG["screen_tstat"])
    for t, arm in sorted(order):
        if t >= threshold:
            continue
        trial = dropped | owned[arm]
        if trial.all():
            continue
        after = evaluate(~trial)
        # `paired` is (whole roster) - (roster without the deletions), so a
        # deletion that helps comes back large and negative and a deletion
        # that costs something comes back large and positive. Keep the
        # deletion unless it costs -- that is `< threshold`, not `> -threshold`.
        if after is None or paired(after) < threshold:
            dropped = trial
    if dropped.all():
        dropped[:] = False
    if torch is not None:
        del panels
        torch.cuda.empty_cache()
    return dropped, verdict


def build_members(inst, X, fit_rows, rows, deadline, keep=None, refit=False):
    """Every estimator on the roster, fitted on `fit_rows`, read out on `rows`.

    `keep` narrows the build to the names still active. After the first fold
    has said which arms are worth their width there is no reason to fit the
    rest again, and on this instrument the difference is the whole clock: the
    widest roster is 208 members and the surviving one is usually a third of
    that. A `+gate` variant needs its base, so the base names are added back.
    """
    names = roster()
    if keep is not None:
        names = [name for name in names if name in keep]
    needed = set(names) | {_base_name(name) for name in names}
    crowd = marginal_member(inst, fit_rows, rows)
    built = {name: crowd for name in names}
    answers = {}

    order = rank_given(inst, fit_rows)
    uses = lambda tag: any(name.endswith("+gate" + tag)
                           or name.startswith("hurdle" + tag + ":")
                           for name in names)
    tables = gate_tables(inst, fit_rows)
    cond = materialise(inst, gate_models(inst, fit_rows), rows) \
        if CONFIG["gate_cond"] and uses("C") else {}
    group = materialise(inst, gate_groups(inst, fit_rows), rows) \
        if CONFIG["gate_group"] and uses("G") else {}
    trees = gate_trees(inst, fit_rows, rows) \
        if CONFIG["gate_tree"] and sklearn_available() and uses("T") else {}
    wanted_k = [k for k in CONFIG["knn_k"] if "knn%d" % k in needed]
    if torch_available() and wanted_k:
        for k, member in zip(wanted_k, knn_member(inst, fit_rows, rows,
                                                  wanted_k, deadline)):
            built["knn%d" % k] = member
    wanted_pos = [w for w in CONFIG["pos_windows"] if "pos%d" % w in needed]
    if CONFIG["pos_members"] and wanted_pos:
        for window, member in zip(wanted_pos,
                                  pos_members(inst, fit_rows, rows,
                                              wanted_pos)):
            built["pos%d" % window] = member
    alphas = list(CONFIG.get("backoff_alphas") or [CONFIG["backoff_alpha"]])
    for slot, alpha in enumerate(alphas):
        for depth in CONFIG["backoff_depths"]:
            if backoff_name(depth, slot) not in needed:
                continue
            if time.time() > deadline:
                break
            built[backoff_name(depth, slot)] = backoff_member(
                inst, fit_rows, rows, order, depth, alpha)
    for classes in CONFIG["regime_classes"]:
        for depth in CONFIG["regime_depths"]:
            tag = "rgm%dd%d" % (classes, depth)
            if tag not in needed or time.time() > deadline:
                continue
            built[tag] = regime_member(inst, fit_rows, rows, order, depth,
                                       classes)
    for index, (blocks, damping) in enumerate(CONFIG["logadd"]):
        if "logadd%d" % index not in needed or time.time() > deadline:
            continue
        built["logadd%d" % index] = logadd_member(inst, fit_rows, rows, order,
                                                  int(blocks), float(damping))
    # v8. TabPFN takes its slice before anything that will spend whatever it
    # is given. Its cost is a known constant -- 56 s for 213 items at 3,500
    # rows -- while the torch families and the tree block both size themselves
    # to the clock that is left, so a family placed behind them is handed a
    # deadline that has already passed and returns the crowd for every item.
    # That is what the first two measurements of this member recorded: an arm
    # that looked like pure width (t = -307) because it had never run.
    #
    # If the arm test drops it after the first fold, that is the only build it
    # is ever fitted in, so the cost of asking is one minute.
    if CONFIG["tabpfn"] and "tabpfn" in needed:
        _t0 = time.time()
        try:
            estimators = int(CONFIG["tabpfn_estimators_refit"]) if refit \
                else int(CONFIG["tabpfn_estimators"])
            cap = float(CONFIG["tabpfn_seconds_refit"]) if refit \
                else float(CONFIG["tabpfn_seconds"])
            member = tabpfn_member(
                inst, fit_rows, rows, min(deadline, time.time() + cap),
                estimators=estimators)
            if member is not None:
                built["tabpfn"] = member
                # v11. The answer-only half for the hurdle members, fitted on
                # the asked rows of every gated item, at the refit's ensemble
                # size when this is the refit.
                wants_hurdle = any(name.startswith("hurdle")
                                   and name.endswith(":tabpfn")
                                   for name in names)
                if CONFIG.get("tabpfn_hurdle") and wants_hurdle:
                    est_answer = int(CONFIG["tabpfn_estimators_refit_answer"]) \
                        if refit else int(CONFIG["tabpfn_estimators"])
                    answer = tabpfn_member(
                        inst, fit_rows, rows,
                        min(deadline, time.time() + cap),
                        estimators=est_answer, answer_only=True,
                        reuse=member)
                    if answer is not None:
                        answers["tabpfn"] = answer
        except Exception:
            pass
        _STAGE["tabpfn"] = _STAGE.get("tabpfn", 0.0) + time.time() - _t0

    if torch_available() and len(fit_rows) >= CONFIG["min_torch_rows"]:
        try:
            ridges = [r for r in CONFIG["ridge_grid"]
                      if "logistic@%g" % r in needed]
            if ridges and time.time() < deadline:
                got = torch_members(inst, X, fit_rows, rows, None,
                                    ridges, deadline)
                for ridge, member in zip(ridges, got):
                    built["logistic@%g" % ridge] = member
            for width in CONFIG["trunk_widths"]:
                ridges = [r for r in CONFIG["trunk_ridges"]
                          if "trunk%d@%g" % (width, r) in needed]
                if not ridges or time.time() > deadline:
                    continue
                got = torch_members(inst, X, fit_rows, rows, width,
                                    ridges, deadline)
                for ridge, member in zip(ridges, got):
                    built["trunk%d@%g" % (width, ridge)] = member
            for width in CONFIG["deep_widths"]:
                ridges = [r for r in CONFIG["deep_ridges"]
                          if "deep%d@%g" % (width, r) in needed]
                if not ridges or time.time() > deadline:
                    continue
                got = torch_members(inst, X, fit_rows, rows, width, ridges,
                                    deadline, layers=CONFIG["deep_layers"])
                for ridge, member in zip(ridges, got):
                    built["deep%d@%g" % (width, ridge)] = member
            classes = [c for c in CONFIG["latent_classes"]
                       if "latent%d" % c in needed]
            if classes and time.time() < deadline:
                got = latent_members(inst, X, fit_rows, rows, classes,
                                     deadline)
                for n_classes, member in zip(classes, got):
                    built["latent%d" % n_classes] = member
            classes = [c for c in CONFIG["latent_joint_classes"]
                       if "latent%dJ" % c in needed]
            if classes and time.time() < deadline:
                got = latent_members(inst, X, fit_rows, rows, classes,
                                     deadline, objective="joint")
                for n_classes, member in zip(classes, got):
                    built["latent%dJ" % n_classes] = member
            # The answer half of the hurdle members: same models, but the
            # sentinel slot dropped and its rows masked out.
            wants = lambda base: any(name.endswith(":" + base)
                                     and name.startswith("hurdle")
                                     for name in names)
            if CONFIG["hurdle"] and time.time() < deadline:
                ridges = [r for r in CONFIG["ridge_grid"]
                          if wants("logistic@%g" % r)]
                if ridges:
                    got = torch_members(inst, X, fit_rows, rows, None, ridges,
                                        deadline, answer_only=True)
                    for ridge, member in zip(ridges, got):
                        answers["logistic@%g" % ridge] = member
                for width in CONFIG["trunk_widths"]:
                    ridges = [r for r in CONFIG["trunk_ridges"]
                              if wants("trunk%d@%g" % (width, r))]
                    if not ridges or time.time() > deadline:
                        continue
                    got = torch_members(inst, X, fit_rows, rows, width,
                                        ridges, deadline, answer_only=True)
                    for ridge, member in zip(ridges, got):
                        answers["trunk%d@%g" % (width, ridge)] = member
                for width in CONFIG["deep_widths"]:
                    ridges = [r for r in CONFIG["deep_ridges"]
                              if wants("deep%d@%g" % (width, r))]
                    if not ridges or time.time() > deadline:
                        continue
                    got = torch_members(inst, X, fit_rows, rows, width,
                                        ridges, deadline, answer_only=True,
                                        layers=CONFIG["deep_layers"])
                    for ridge, member in zip(ridges, got):
                        answers["deep%d@%g" % (width, ridge)] = member
        except Exception:            # no GPU, no memory, no torch: the counting
            pass                     # estimators still stand on their own.

    # v5. The trees. Fitted after the GPU families because they are the ones
    # that can be truncated item-wise without breaking anything: a member the
    # clock cut short keeps the crowd marginal on the items it never reached,
    # and the per-item stack weights those items on the other families.
    if sklearn_available():
        # One clock for the whole tree block, not one per call: the answer-only
        # half is the same 213 fits again, and a cap that each of them reads
        # separately is a cap on nothing.
        stop = min(deadline, time.time() + CONFIG["gbm_share"]
                   * max(deadline - time.time(), 0.0))
        for index, params in enumerate(CONFIG["gbm"]):
            slots = [k for k in range(len(gbm_floor_list()))
                     if gbm_name(index, k) in needed]
            if not slots:
                continue
            for slot, member in enumerate(gbm_member(inst, fit_rows, rows,
                                                     params, stop)):
                if slot in slots:
                    built[gbm_name(index, slot)] = member
        if CONFIG["gbm_hurdle"]:
            for index, params in enumerate(CONFIG["gbm"]):
                if not any(name.startswith("hurdle")
                           and name.endswith(":gbm%d" % index)
                           for name in names):
                    continue
                answers["gbm%d" % index] = gbm_member(
                    inst, fit_rows, rows, params, stop, answer_only=True)[0]

    # v7. The hub family reads P(hub | x) off the members already built in
    # this same build, so it is exactly as out-of-fold as they are. The
    # distribution it reads is the mean of the strongest families rather than
    # any one of them -- the hub is a three-way question carrying 73 items
    # behind it, and a single family's opinion of it is the wrong thing to
    # multiply 73 tables by.
    if CONFIG["hub"] and "hub" in needed:
        # v9. `tabpfn` joins the sources. The hub multiplies 73 tables by one
        # three-way distribution, so what that distribution is made of is most
        # of the family; on `fit_cell` the arm test gives `tabpfn+gateC` the
        # top per-item weight of the whole roster, and reading the hub off
        # everything except the strongest member was an accident of order.
        source = [n for n in ("tabpfn", "gbm0",
                              backoff_name(max(CONFIG["backoff_depths"]), 0),
                              "trunk64@0.05", "logistic@0.03")
                  if n in built and built[n] is not crowd]
        if source:
            blend = [sum(built[n][j] for n in source) / len(source)
                     for j in range(len(inst.targets))]
        else:
            blend = crowd
        try:
            member = hub_member(inst, fit_rows, rows, order, blend)
            if member is not None:
                built["hub"] = member
        except Exception:
            pass

    # v7. Single precision before the gate suffixes multiply the roster by
    # four. The widest UNHCR roster is 198 members read out over 3,381 rows
    # and 213 items; in float64 that is 4.5 GB of members alone, against a
    # container the grader kills at 16 GB. Converting here rather than on the
    # way out means the doubled copy never exists: everything `gate_refine`
    # and `hurdle_member` build downstream is single precision already.
    crowd32 = [np.ascontiguousarray(v, np.float32) for v in crowd]
    for name in list(built):
        member = built[name]
        built[name] = crowd32 if member is crowd else \
            [np.ascontiguousarray(v, np.float32) for v in member]
    for name in list(answers):
        answers[name] = [np.ascontiguousarray(v, np.float32)
                         for v in answers[name]]
    crowd = crowd32

    for name in names:
        if "^" in name and not name.endswith("+gate"):
            base, gamma = name.rsplit("^", 1)
            built[name] = sharpen(built[base], float(gamma))

    sentinels, sentinels_cond = {}, {}
    sentinels_group, sentinels_tree = {}, {}
    for name in names:
        if name.endswith("+gateT"):
            base = name[:-6]
            built[name], sentinels_tree[base] = gate_refine(
                inst, built[base], rows, trees, conditional=True)
        elif name.endswith("+gateG"):
            base = name[:-6]
            built[name], sentinels_group[base] = gate_refine(
                inst, built[base], rows, group, conditional=True)
        elif name.endswith("+gateC"):
            base = name[:-6]
            built[name], sentinels_cond[base] = gate_refine(
                inst, built[base], rows, cond, conditional=True)
        elif name.endswith("+gate"):
            base = name[:-5]
            built[name], sentinels[base] = gate_refine(inst, built[base], rows,
                                                       tables)
    for name in names:
        if name.startswith("hurdleT:"):
            base, measured, suffix = name[8:], sentinels_tree, "+gateT"
        elif name.startswith("hurdleG:"):
            base, measured, suffix = name[8:], sentinels_group, "+gateG"
        elif name.startswith("hurdleC:"):
            base, measured, suffix = name[8:], sentinels_cond, "+gateC"
        elif name.startswith("hurdle:"):
            base, measured, suffix = name[7:], sentinels, "+gate"
        else:
            continue
        refined = built.get(base + suffix, crowd)
        if base in answers and base in measured:
            built[name] = hurdle_member(inst, answers[base], measured[base],
                                        refined)
        else:
            built[name] = refined

    # A vector of the wrong width fails the submission outright, and the day's
    # one attempt with it. Anything that does not match the schema is replaced
    # by the crowd, which is always the right shape.
    for name in names:
        member = built[name]
        for j, item in enumerate(inst.targets):
            if np.shape(member[j])[1] != inst.width[item]:
                built[name] = crowd
                break
    # v7. Single precision on the way out. The widest roster reads 208 members
    # over three thousand rows and 213 items, which is 4.7 GB in float64 and
    # 2.4 in float32, against a container that is killed at 16 GB. Members
    # that are the same object -- everything the clock cut short is the crowd
    # -- are converted once.
    out = []
    for name in names:
        member = built[name]
        out.append(member if member is crowd or member[0].dtype == np.float32
                   else [np.ascontiguousarray(v, np.float32)
                         for v in member])
    return out, names


def predict(frame, schema):
    # CONFIG is module state and the three instruments share a process, so the
    # per-instrument block has to be undone before the next call rather than
    # accumulated.
    global _BASE
    if _BASE is None:
        _BASE = dict(CONFIG)
    CONFIG.clear()
    CONFIG.update(_BASE)

    inst = Instrument(frame, schema)
    tuned = CONFIG["per_instrument"].get(inst.key)
    if tuned:
        CONFIG.update(tuned)
        CONFIG.update(_OVERRIDE)
        inst = Instrument(frame, schema)
    if inst.phase == 2 and CONFIG["phase2_overrides"]:
        CONFIG.update(CONFIG["phase2_overrides"])
        # An explicit override -- `SB_CONFIG` locally, `package.py --set` on a
        # submission -- has to win over the phase block, or a phase-2 ablation
        # cannot be run at all.
        CONFIG.update(_OVERRIDE)
        inst = Instrument(frame, schema)     # cheap, and the knobs may change it
    global _CALLS
    call = _CALLS
    _CALLS += 1

    # v8. Settled by the organisers: 900 s covers all three instruments
    # together. That is the starter kit's `DOCUMENTATION.md` reading, and the
    # site's "15 minutes per instrument" is wrong. The code already assumed
    # the narrow one -- an instrument takes its share of the budget and the
    # three of them together never pass it, measured from import -- so nothing
    # here changes.
    #
    # What does change is the *final* phase. The same page that was wrong
    # about development says "60 minutes per instrument" about the final, so
    # `phase2_budget` is 3,600 s for the three of them together as well.
    # Exceeding it fails the one submission the prize is decided on, and a
    # phase-2 configuration is only allowed to ship after an `--all` run
    # measures the three together inside it.
    #
    # The shares are not equal because the leaderboard is not: UNICEF and
    # World Bank are saturated -- every team within 0.01 of every other -- and
    # UNHCR carries all of the spread, so an equal split spends two thirds of
    # the clock where no score can move. With one shared budget, the shares
    # are the whole of the allocation decision.
    # v9. Three things were leaving the clock unspent, and together they were
    # a quarter of it (measured: 641 s of 900 on the deployed frames).
    #
    #  (a) The origin. The budget was counted from module import, so the
    #      container's prologue -- interpreter, torch, the first frame -- was
    #      free time in the arithmetic and charged for again in `safety`. It
    #      is now counted from the process's own start.
    #  (b) The shares summed to 0.96, so 4% was allocated to nobody.
    #  (c) The shares were caps rather than entitlements. UNICEF and World
    #      Bank both finish under theirs, and what they left went nowhere,
    #      because UNHCR's deadline was `share x budget` and not "your share
    #      of what is still on the clock". The three calls run in turn inside
    #      one process, so the remainder is knowable: divide what is left by
    #      the shares of the instruments that have *not* been called yet.
    #
    # (c) is also safe under the reading this code has always defended
    # against -- one container per instrument. Then nothing has been called
    # yet, the denominator is the whole share vector, and the deadline is
    # exactly `share x budget` again.
    budget = CONFIG["phase%d_budget" % inst.phase] * CONFIG["safety"]
    shares = CONFIG["budget_share"]
    default = 1.0 / max(CONFIG["instruments"], 1)
    share = shares.get(inst.key, default)
    ahead = sum(v for k, v in shares.items() if k not in _SEEN)
    if inst.key not in shares:
        ahead += default
    together = max(_STARTED_AT + budget - time.time(), 10.0)
    entitled = together * share / max(ahead, share, 1e-6)
    call_deadline = time.time() + max(min(entitled, together), 10.0)
    _SEEN.add(inst.key)

    _STAGE.update(start=time.time(), folds=0.0, refit=0.0, screen=0.0,
                  tabpfn=0.0, calib0=time.time(), deadline=call_deadline)

    X = design(inst)
    Y = inst.target_codes()
    n_fit, n_items = len(inst.fit_rows), len(inst.targets)

    verdict = {}
    shed, footprint = fit_roster_to_memory(
        inst, len(inst.fit_rows) // max(2, int(CONFIG["folds"]))
        + len(inst.out_rows))
    names = roster()

    # ---- out-of-fold likelihoods, which is what chooses the weights -------
    #
    # Each fold's models also read out the rows that are actually being
    # predicted, so the same models that earned the weights are the ones the
    # weights are applied to, and averaging over the folds is a bag: less
    # variance than any single fit, at no extra fitting cost.
    folds = max(2, int(CONFIG["folds"]))
    passes = max(1, int(CONFIG.get("fold_passes", 1)))
    rng = np.random.default_rng(CONFIG["seed"])
    assignment = rng.permutation(n_fit) % folds
    likelihood = np.zeros((len(names), n_fit, n_items), np.float32)
    # v10. How many folds have read each row out for each member. A pass is a
    # whole cover, so after one pass this is 1 everywhere for every member the
    # screen kept, and a second pass makes it 2 -- which is a bag at the cell
    # level, the same object the shipped vectors are. Members the screen drops
    # keep the single cover the first fold gave them and are excluded from the
    # mixture anyway.
    cover = np.zeros((len(names), n_fit), np.float32)
    bag = [[np.zeros((len(inst.out_rows), inst.width[name]), np.float32)
            for name in inst.targets] for _ in names]
    bag_weight = np.zeros(len(names))
    slot_of = {name: i for i, name in enumerate(names)}
    active = np.ones(len(names), bool)
    room0 = call_deadline - CONFIG["calib_reserve"] * share * budget
    fold_share = float(CONFIG["fold_share"])
    fold_end = time.time() + fold_share * max(room0 - time.time(), 10.0)
    done, cost = 0, 0.0

    # A slice of the out-of-fold cells kept in full, not just at the answer:
    # `calibrate` needs the whole vector to renormalise a sharpened one, and
    # keeping every out-of-fold vector for every member would be gigabytes.
    # v7. The first fold is the wide one and it is the one the arm test reads,
    # so the vectors kept for the calibration stage start with the fold after
    # it -- otherwise the stack would be ragged across members that the first
    # fold built and the rest did not.
    collect_from = 1 if CONFIG["screen"] else 0
    per_fold = max(0, int(CONFIG["calib_rows"])
                   // max(folds - collect_from, 1)) \
        if CONFIG["calibrate"] else 0
    kept = [[] for _ in names]
    kept_truth = []
    kept_rows = []
    # v11. Raking weights on the visible rows (mean 1), or None.
    row_weight = shift_weights(inst)
    weight_by_row = np.ones(inst.n)
    if row_weight is not None:
        weight_by_row[inst.fit_rows] = row_weight
    # v10. The target is a number of rows, not a number of folds. With passes
    # the loop may run more folds than a single cover, and the slice stops
    # growing when it has what `calib_rows` asked for.
    calib_target = int(CONFIG["calib_rows"]) if CONFIG["calibrate"] else 0
    calib_have = 0

    # v7. A fold is worth about +0.0013 and a truncated refit is worth much
    # less than that -- the refit is the only build fitted on every visible
    # row. So the loop stops when the fold it is about to start would not
    # leave room for a complete one, estimated from what a fold just cost
    # (the refit sees 1/(1-1/folds) times the rows, so it costs about that
    # much more). `fold_share` stays as a ceiling.
    # v9. Measured rather than derived. The derived number was the ratio of
    # training rows -- a refit sees 1/(1-1/folds) times as many as a fold --
    # and at four folds that is 1.33, which refused the fourth fold on UNHCR
    # by six seconds. It is the wrong ratio: a fold also reads out its own
    # held-out block on top of the rows being predicted (5,451 rows against
    # the refit's 3,381), and that nearly cancels the extra training rows.
    # Measured on the delivered shapes the refit costs 1.05-1.15 of the fold
    # before it (UNHCR 120.3 s against 110 s, and 109.2 s against 104 s in
    # v8), so 1.15 is the measurement with the margin on the safe side. An
    # underestimate is not fatal either way -- the refit carries its own
    # deadline and truncates -- but a fold refused for nothing is a fold lost.
    refit_scale = float(CONFIG["refit_scale"])
    stop = False
    for turn in range(passes):
        if stop:
            break
        if turn:
            # A fresh partition, so the second cover is not the first one
            # again. Its cells average into the first at the same rows.
            assignment = rng.permutation(n_fit) % folds
        for fold in range(folds):
            # v10. One test, not three. `fold_share` has already carved the
            # refit's budget out of the window, so the old refit reservation
            # was charging for it twice -- and charging with `cost`, which is
            # the last fold's *allocation* rather than its cost, because every
            # family here expands to fill the deadline it is handed. What is
            # left is the honest question: is there room for another fold
            # inside the fold budget? Half a fold is the threshold because a
            # fold that starts is a fold that truncates itself gracefully --
            # its members fall back to the crowd marginal, which is a member
            # the stack already weights.
            if n_fit < CONFIG["min_fold_rows"] \
                    or (done and time.time() + 0.5 * cost > fold_end):
                stop = True
                break
            started = time.time()
            held = assignment == fold
            inner_out = inst.fit_rows[held]
            readout = np.concatenate([inner_out, inst.out_rows])
            # v5. Each fold gets its own slice of the fold budget. Handing
            # every fold the same global deadline was harmless while every
            # member was cheap; a member that will spend whatever it is given
            # turns it into "fold 0 takes the lot", and folds are what the
            # weights are made of.
            # v10. Except the first one, which builds every screen arm while
            # the rest build what the screen kept, and which the arm test is
            # read from. An even share truncates it, and a truncated first
            # fold does not cost a fold -- it costs the roster, because the
            # arms it could not finish are judged on a build that is partly
            # the crowd marginal.
            width_scale = float(CONFIG.get("wide_fold_scale", 1.0)) \
                if (turn == 0 and fold == 0 and CONFIG["screen"]) else 1.0
            slice_end = min(fold_end, started + max(
                width_scale * CONFIG["fold_slack"]
                * (fold_end - started) / max(folds - fold, 1), 5.0))
            keep = {names[i] for i in np.flatnonzero(active)}
            members, got = build_members(inst, X, inst.fit_rows[~held],
                                         readout, slice_end, keep=keep)
            truth = Y[inner_out]
            index = np.arange(len(inner_out))
            collecting = done >= collect_from and per_fold > 0 \
                and calib_have < calib_target
            take = (index if 0 < len(index) <= per_fold else
                    rng.choice(len(index), per_fold, replace=False)) \
                if collecting else None
            for local, name in enumerate(got):
                m = slot_of[name]
                member = members[local]
                row = []
                for j in range(n_items):
                    likelihood[m, held, j] += member[j][
                        index, np.maximum(truth[:, j], 0)]
                    bag[m][j] += member[j][len(inner_out):]
                    if take is not None:
                        row.append(member[j][take].astype(np.float16))
                if take is not None:
                    kept[m].append(row)
                cover[m, held] += 1.0
                bag_weight[m] += 1.0
            if take is not None:
                kept_truth.append(np.maximum(truth[take], 0))
                kept_rows.append(inner_out[take])
                calib_have += len(take)
            del members
            # v7. The first fold is the wide one; after the arm test the
            # roster is a third of it, so the cost of the fold just finished
            # predicts the next one and the cost of the widest one does not.
            # Taking the max here is what cost the third fold.
            cost = time.time() - started
            done += 1

            # ---- the arm test, once, on the first fold's cells -------------
            if done == 1 and CONFIG["screen"] and folds * passes > 1:
                _screen0 = time.time()
                try:
                    cells = likelihood[:, held, :]
                    if CONFIG["floor_aware"]:
                        f = np.float32(CONFIG["grader_floor"])
                        widths = np.array(
                            [inst.width[t] for t in inst.targets], np.float32)
                        cells = cells * (1.0 - widths * f) + f
                    dropped, verdict = prune_arms(names, cells, n_fit,
                                                  CONFIG["seed"])
                    active &= ~dropped
                    del cells
                except Exception:
                    verdict = {}
                _STAGE["screen"] = time.time() - _screen0

    # ---- refit on everything visible, and pool -----------------------------
    # The refit is the build that matters most -- it is the only one fitted on
    # every visible row -- but the calibration stages that follow it read every
    # member's held-out vectors and are not interruptible, so they are paid
    # for before the refit starts rather than out of whatever is left.
    refit_end = call_deadline - CONFIG["calib_reserve"] * share * budget
    _STAGE["folds"] = time.time() - _STAGE["start"]
    _refit0 = time.time()
    keep = {names[i] for i in np.flatnonzero(active)}
    members, got = build_members(inst, X, inst.fit_rows, inst.out_rows,
                                 max(refit_end, time.time() + 5.0), keep=keep,
                                 refit=True)
    full = CONFIG["bag_full_weight"]
    for local, name in enumerate(got):
        m = slot_of[name]
        for j in range(n_items):
            bag[m][j] += full * members[local][j]
        bag_weight[m] += full
    del members
    for m in range(len(names)):
        if bag_weight[m] <= 0:
            continue
        for j in range(n_items):
            bag[m][j] /= bag_weight[m]
    _STAGE["refit"] = time.time() - _refit0

    if done:
        # v10. A row is usable if some fold read it out, and its cell is the
        # mean over the folds that did -- which is the same bag the shipped
        # vectors are, taken at the answer. The pass loop guarantees this is a
        # whole cover after the first pass, so `seen` is normally everything.
        seen = cover.max(axis=0) > 0
        cells = likelihood[:, seen, :] / np.maximum(cover[:, seen], 1.0)[
            :, :, None]
        if CONFIG["floor_aware"]:
            # sum_m w_m (f + c p_m) = f + c sum_m w_m p_m, so this is the
            # grader's floored pool and EM keeps maximising the real metric.
            f = float(CONFIG["grader_floor"])
            widths = np.array([inst.width[t] for t in inst.targets], np.float32)
            cells *= (1.0 - widths * f).astype(np.float32)
            cells += np.float32(f)
        live = np.flatnonzero(active & (bag_weight > 0))
        if live.size < 2:
            live = np.arange(len(names))
        panel = cells[live]
        rw_seen = row_weight[seen] if row_weight is not None else None
        shared = em_weights(panel.reshape(live.size, -1, 1), live.size, 1,
                            row_weight=(np.repeat(rw_seen, n_items)
                                        if rw_seen is not None else None))[0]
        grid = list(CONFIG["em_pool_grid"]) or [float(CONFIG["em_pool"])]
        chosen = float(CONFIG["em_pool"])
        if len(grid) > 1 and panel.shape[1] >= 800:
            try:
                shuffle = np.random.default_rng(
                    CONFIG["seed"] + 13).permutation(panel.shape[1])
                middle = shuffle.size // 2
                pairs = ((shuffle[:middle], shuffle[middle:]),
                         (shuffle[middle:], shuffle[:middle]))
                keep_pool, best = CONFIG["em_pool"], -np.inf
                for candidate in grid:
                    CONFIG["em_pool"] = candidate
                    total = 0.0
                    for fit_half, read_half in pairs:
                        w = em_weights(panel[:, fit_half, :], live.size,
                                       n_items, shared,
                                       row_weight=(rw_seen[fit_half]
                                                   if rw_seen is not None
                                                   else None))
                        got = np.einsum("jm,mnj->nj", w,
                                        panel[:, read_half, :])
                        logs = np.log(np.maximum(got, _EPS))
                        if rw_seen is not None:
                            logs = logs * rw_seen[read_half][:, None]
                        total += float(logs.sum())
                    if total > best:
                        best, chosen = total, float(candidate)
                CONFIG["em_pool"] = keep_pool
            except Exception:
                chosen = float(CONFIG["em_pool"])
        CONFIG["em_pool"] = chosen
        weights = np.zeros((n_items, len(names)))
        weights[:, live] = em_weights(panel, live.size, n_items, shared,
                                      row_weight=rw_seen)
        del panel
    else:
        weights = np.full((n_items, len(names)), 1.0 / len(names))
    del likelihood
    coverage = float((cover.max(axis=0) > 0).mean())
    passes_done = float(cover.max()) if cover.size else 0.0
    del cover

    _STAGE["calib0"] = time.time()

    # ---- one exponent on the pool, fitted on the cells it did not see ------
    #
    # A linear pool sits between its members, so it is flatter than any of
    # them, and the roster is nearly fifty members wide. How much that costs is
    # a property of the instrument, not something to assume: this measures it
    # on the held-out cells and takes the exponent that scored best there,
    # pulled back towards 1 because those cells were read out by fold models
    # and the shipped numbers come from a bag that also holds the full refit.
    # A grid containing 1.0 means the answer can be "nothing to fix".
    grid = np.asarray(CONFIG["calib_grid"], float)
    gamma = np.ones(n_items)
    gate_slope, gate_offset = np.ones(n_items), np.zeros(n_items)
    calib_members = [m for m in range(len(names)) if kept[m]]
    shrink, calib_used = float(CONFIG["calib_shrink"]), 0
    rule = np.zeros(n_items, np.int64)
    if kept_truth and done and calib_members:
        truth_all = np.concatenate(kept_truth, axis=0)
        cell_weight = weight_by_row[np.concatenate(kept_rows, axis=0)]
        held_weight = weights[:, calib_members]
        seat = np.arange(len(truth_all))
        curves = np.zeros((n_items, grid.size))
        # v7. Two halves of the held-out rows, kept apart so that how far to
        # trust the fitted exponent is itself something the data says. The
        # pooled vectors are kept -- 213 items by 1,800 rows is twelve
        # megabytes, and everything after this is arithmetic on them.
        cut = np.random.default_rng(CONFIG["seed"] + 11).permutation(
            len(truth_all))
        halves = (cut[:len(cut) // 2], cut[len(cut) // 2:])
        rules = ["linear", "log"] if CONFIG["pool_rule"] == "auto" \
            else [CONFIG["pool_rule"]]
        width_grid = grid.size * len(rules)
        curves = np.zeros((n_items, width_grid))
        parts = [np.zeros((n_items, width_grid)) for _ in halves]
        pools, floors = [], []
        for j in range(n_items):
            stack = np.stack([np.concatenate([fold[j] for fold in kept[m]],
                                             axis=0)
                              for m in calib_members], axis=0).astype(
                                  np.float32)
            candidates = []
            for rule in rules:
                if rule == "linear":
                    pool = np.einsum("m,mnk->nk", held_weight[j], stack)
                else:
                    pool = np.exp(np.einsum(
                        "m,mnk->nk", held_weight[j],
                        np.log(np.maximum(stack, 1e-9))))
                    pool /= np.maximum(pool.sum(axis=1, keepdims=True), _EPS)
                candidates.append(np.maximum(pool, _EPS))
            fj = float(CONFIG["grader_floor"]) if CONFIG["floor_aware"] else 0.0
            cj = 1.0 - candidates[0].shape[1] * fj
            pools.append(candidates)
            floors.append((fj, cj))
            for r, pool in enumerate(candidates):
                for g, power in enumerate(grid):
                    sharp = pool ** power
                    sharp /= sharp.sum(axis=1, keepdims=True)
                    sharp = fj + cj * sharp
                    at = np.log(np.maximum(sharp[seat, truth_all[:, j]], _EPS))
                    at = at * cell_weight
                    curves[j, r * grid.size + g] = at.sum()
                    for h, half in enumerate(halves):
                        parts[h][j, r * grid.size + g] = at[half].sum()

        def apply_gamma(j, rows, rule, power):
            """Log score of item j's pooled vector, sharpened, on `rows`."""
            fj, cj = floors[j]
            sharp = pools[j][rule][rows] ** power
            sharp /= sharp.sum(axis=1, keepdims=True)
            sharp = fj + cj * sharp
            return float((np.log(np.maximum(
                sharp[np.arange(rows.size), truth_all[rows, j]], _EPS))
                * cell_weight[rows]).sum())

        # An item's own held-out cells decide its exponent, but only against a
        # prior worth `calib_prior` items' worth of the instrument-wide curve,
        # so a 900-cell slice cannot pick a corner of the grid on noise.
        shrink = float(CONFIG["calib_shrink"])
        if len(halves[0]) > 40 and CONFIG.get("calib_shrink_grid"):
            scores = []
            for candidate in CONFIG["calib_shrink_grid"]:
                total = 0.0
                for h, half in enumerate(halves):
                    other = halves[1 - h]
                    fit = parts[1 - h]
                    cell = np.argmax(
                        fit + CONFIG["calib_prior"] * fit.mean(axis=0)[None, :],
                        axis=1)
                    for j in range(n_items):
                        total += apply_gamma(
                            j, half, int(cell[j]) // grid.size,
                            1.0 + candidate
                            * (grid[int(cell[j]) % grid.size] - 1.0))
                scores.append(total)
            shrink = float(CONFIG["calib_shrink_grid"][int(np.argmax(scores))])
        shared = curves.mean(axis=0)
        combined = curves + CONFIG["calib_prior"] * shared[None, :]
        cell = np.argmax(combined, axis=1)
        rule = (cell // grid.size).astype(np.int64)
        best = grid[cell % grid.size]
        gamma = 1.0 + shrink * (best - 1.0)
        calib_used = len(truth_all)

        # ---- and a slope and an offset on P(sentinel), for the gated items --
        #
        # The exponent moves a whole vector at once. On this instrument most
        # of a cell's information is a single number -- whether the cell is
        # blank at all -- and that number is the composition of a predicted
        # parent, a measured gate fidelity and a linear pool, each of which
        # pulls it towards the middle. Fitted the same way as the exponent, on
        # the same held-out cells, against the same instrument-wide prior, and
        # with (slope 1, offset 0) in the grid so it can decline.
        if CONFIG["calib_gate"]:
            slopes = np.asarray(CONFIG["calib_gate_slopes"], float)
            offsets = np.asarray(CONFIG["calib_gate_offsets"], float)
            pairs = [(b, a) for b in slopes for a in offsets]
            curves = np.full((n_items, len(pairs)), -np.inf)
            gated = [j for j, name in enumerate(inst.targets)
                     if inst.gate[name]]
            for j in gated:
                chosen = pools[j][int(rule[j])]
                pool = chosen ** gamma[j] if gamma[j] != 1.0 else chosen.copy()
                pool /= pool.sum(axis=1, keepdims=True)
                share = np.clip(pool[:, -1], 1e-6, 1 - 1e-6)
                body = pool[:, :-1]
                body = body / np.maximum(body.sum(axis=1, keepdims=True), _EPS)
                logit = np.log(share / (1.0 - share))
                answer = truth_all[:, j]
                is_sentinel = answer == pool.shape[1] - 1
                body_at = np.log(np.maximum(
                    body[seat, np.minimum(answer, body.shape[1] - 1)], _EPS))
                fj = float(CONFIG["grader_floor"]) \
                    if CONFIG["floor_aware"] else 0.0
                cj = 1.0 - pool.shape[1] * fj
                body_p = np.maximum(
                    body[seat, np.minimum(answer, body.shape[1] - 1)], _EPS)
                for g, (slope, offset) in enumerate(pairs):
                    moved = 1.0 / (1.0 + np.exp(-(offset + slope * logit)))
                    moved = np.clip(moved, 1e-9, 1 - 1e-9)
                    at = np.where(is_sentinel, moved, (1.0 - moved) * body_p)
                    curves[j, g] = (np.log(np.maximum(fj + cj * at, _EPS))
                                    * cell_weight).sum()
            if gated:
                shared = curves[gated].mean(axis=0)
                combined = curves + CONFIG["calib_prior"] * shared[None, :]
                pick = np.argmax(combined, axis=1)
                for j in gated:
                    slope, offset = pairs[int(pick[j])]
                    gate_slope[j] = 1.0 + shrink * (slope - 1.0)
                    gate_offset[j] = shrink * offset
        del kept, kept_truth, kept_rows, pools

    if os.environ.get("SB_DEBUG"):
        print("[debug] stages: screen %.1fs  folds %.1fs  refit %.1fs  "
              "calib %.1fs" % (_STAGE["screen"], _STAGE["folds"],
                               _STAGE["refit"],
                               time.time() - _STAGE["calib0"]))
        # v9. The clock, in the units the grader uses. `slack` is what is left
        # of the call's own deadline (negative is overshoot, which is the one
        # number `safety` has to cover) and `wall` is time since this process
        # started, which is what the 900 s is measured against.
        print("[debug] clock: prologue %.1fs  deadline %.1fs  used %.1fs  "
              "slack %+.1fs  wall %.1fs / %.0fs"
              % (_PROLOGUE, _STAGE["deadline"] - _STAGE["start"],
                 time.time() - _STAGE["start"],
                 _STAGE["deadline"] - time.time(),
                 time.time() - _STARTED_AT,
                 CONFIG["phase%d_budget" % inst.phase]))
        print("[debug] tabpfn %.1fs total across builds" % _STAGE.get("tabpfn", 0.0))
        # v10. The two numbers v9 shipped without: how many folds actually
        # ran, and what share of the delivered rows carry an out-of-fold cell
        # for the EM to read. Coverage below 1.0 is the failure v9 shipped.
        print("[debug] folds %d done (%d blocks x up to %d passes)  "
              "coverage %.3f  max passes on a row %.0f  fold_share %.4f"
              % (done, folds, passes, coverage, passes_done, fold_share))
        print("[debug] roster footprint %.2f GB, shed %s"
              % (footprint, shed or "nothing"))
        if verdict:
            print("[debug] screen (t, members), kept at t>=%.1f:"
                  % CONFIG["screen_tstat"])
            for arm, (t, count) in sorted(
                    verdict.items(), key=lambda kv: -(kv[1][0] if kv[1][0]
                                                      == kv[1][0] else -99)):
                mark = "KEEP" if t >= CONFIG["screen_tstat"] else "    "
                print("[debug]   %-14s t=%+8.2f  n=%3d  %s"
                      % (arm, t, count, mark))
        print("[debug] em_pool %.0f" % CONFIG["em_pool"])
        print("[debug] shift: columns %s  ess %s"
              % (_STAGE.get("shifted", []),
                 ("%.0f" % _STAGE["shift_ess"]) if "shift_ess" in _STAGE
                 else "n/a (weights are 1)"))
        print("[debug] calib shrink %.2f  rows %d  members %d"
              % (shrink, calib_used, len(calib_members)))
        print("[debug] log pool chosen for %d/%d items"
              % (int((rule > 0).sum()), n_items))
        print("[debug] gamma mean %.3f  median %.3f  [%.2f, %.2f]"
              % (gamma.mean(), float(np.median(gamma)), gamma.min(),
                 gamma.max()))
        print("[debug] gate slope median %.3f  offset median %+.3f"
              % (float(np.median(gate_slope)), float(np.median(gate_offset))))
        top = weights.mean(axis=0)
        order = np.argsort(-top)[:10]
        print("[debug] folds=%d  roster %d wide -> %d live  bag %.1f"
              % (done, len(names), int((active & (bag_weight > 0)).sum()),
                 float(bag_weight.max())))
        print("[debug] " + "  ".join("%s=%.3f" % (names[i], top[i]) for i in order))

    # v6. With SB_DUMP_BAG set, hand the whole bag out to the caller: every
    # member's vectors on the rows being predicted, the weights EM chose, and
    # the fitted calibration. Everything after this point -- pooling,
    # exponent, sentinel slope -- is arithmetic on these arrays, so a probe
    # that holds them can try a different pooling scheme in a second instead
    # of paying six minutes to refit the whole roster to find out.
    if os.environ.get("SB_DUMP_BAG"):
        _BAG.clear()
        _BAG.update(bag=bag, weights=weights, names=names,
                    targets=list(inst.targets), gamma=gamma,
                    gate_slope=gate_slope, gate_offset=gate_offset,
                    gated=[bool(inst.gate[n]) for n in inst.targets],
                    out_rows=np.asarray(inst.out_rows), folds=done)

    pooled = {}
    live = np.flatnonzero(weights.max(axis=0) > 0)
    if live.size == 0:
        live = np.arange(len(names))
    for j, name in enumerate(inst.targets):
        stack = np.stack([bag[m][j] for m in live], axis=0)
        if rule[j] == 0:
            vector = np.einsum("m,mnk->nk", weights[j, live], stack)
        else:
            vector = np.exp(np.einsum("m,mnk->nk", weights[j, live],
                                      np.log(np.maximum(stack, 1e-9))))
            vector /= np.maximum(vector.sum(axis=1, keepdims=True), _EPS)
        vector = np.asarray(vector, np.float64)
        if gamma[j] != 1.0:
            vector = np.maximum(vector, _EPS) ** gamma[j]
        vector = vector / np.maximum(vector.sum(axis=1, keepdims=True), _EPS)
        if inst.gate[name] and (gate_slope[j] != 1.0 or gate_offset[j] != 0.0):
            share = np.clip(vector[:, -1], 1e-6, 1 - 1e-6)
            logit = np.log(share / (1.0 - share))
            moved = 1.0 / (1.0 + np.exp(
                -(gate_offset[j] + gate_slope[j] * logit)))
            moved = np.clip(moved, 1e-9, 1 - 1e-9)
            body = vector[:, :-1]
            body = body / np.maximum(body.sum(axis=1, keepdims=True), _EPS)
            vector = np.concatenate(
                [body * (1.0 - moved)[:, None], moved[:, None]], axis=1)
        pooled[name] = vector / np.maximum(
            vector.sum(axis=1, keepdims=True), _EPS)

    # ---- one vector per blank cell, canonical order ------------------------
    fallback = {}
    for name in inst.items:
        if name in pooled:
            continue
        counts = np.bincount(inst.code[name][inst.fit_rows].clip(min=0),
                             minlength=inst.width[name]).astype(np.float64)
        counts += CONFIG["smoothing"]
        fallback[name] = counts / counts.sum()

    place = np.full(inst.n, -1, np.int64)
    place[inst.out_rows] = np.arange(len(inst.out_rows))
    # The graded cells are the held-out rows crossed with the PREDICT items:
    # score.py blanks the scored items and nothing else, and a GIVEN cell is
    # visible for everybody, so a blank can only be a scored one. PREDICT items
    # keep their schema order, so np.nonzero walking row-major is the canonical
    # order exactly: rows top to bottom, items in schema key order within a row.
    blank = np.stack([inst.isna[name] for name in inst.targets], axis=1)
    rows, columns = np.nonzero(blank)

    vectors = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        name = inst.targets[column]
        seat = place[row]
        if name in pooled and seat >= 0:
            vector = pooled[name][seat]
        else:
            vector = fallback.get(name)
            if vector is None:
                vector = np.full(inst.width[name], 1.0 / inst.width[name])
        vector = np.asarray(vector, dtype=float)
        if not np.isfinite(vector).all() or not vector.sum() > 0:
            vector = np.maximum(np.nan_to_num(vector, nan=0.0, posinf=0.0,
                                              neginf=0.0), 0.0)
            if not vector.sum() > 0:
                vector = np.full(vector.size, 1.0 / max(vector.size, 1))
        vectors.append(np.maximum(vector, 0.0))
    return vectors
