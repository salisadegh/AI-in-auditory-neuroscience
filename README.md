# Label construction and detector rankings in objective ABR detection

Analysis code and expert-annotation mappings for the study *"Label construction
can determine measured detector performance on small auditory brainstem
response benchmarks."*

Two small public corpora recur as evaluation data for objective auditory
brainstem response (ABR) detection. Neither distributes expert annotations by
default, so studies often substitute an automatically computed proxy — usually a
two-replication reproducibility criterion. This repository tests what that
substitution costs, using audiologist annotations published as supplementary
material to a recent multicentre study and mapped here onto the corpus records.

## What the analysis shows

* The reproducibility proxy agrees with expert judgement at Cohen's κ = 0.17
  (tone-pip corpus) and κ = 0.25 (click corpus), against a median κ of 0.69
  between six audiologists annotating the same recordings.
* Changing only the label changes which detector ranks first. Relative to an
  untrained instrument statistic the interaction survives condition-level
  analysis and an exact permutation test; between two trained detectors it does
  not, which is itself the point about corpora of this size.
* Re-scoring the same detector against each of six annotators moves AUROC by up
  to 0.11 — comparable to the differences between detectors and to the smallest
  difference the design can resolve.
* Two label-free baselines are strong: stimulus level alone reaches AUROC 0.93
  within 1 kHz, exceeding every waveform-based detector at that frequency.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download the corpora as described in data/README.md
python run_all.py
```

Runs on CPU in a few minutes. Results are written to `results/` as JSON and
figures to `figures/` as 300-dpi PNG plus vector PDF.

## Layout

```
labels/     expert annotations mapped onto corpus records (the reusable artefact)
src/        analysis modules
  config.py            paths, windows, filter band, seed
  features.py          conditioning and the six scale-invariant descriptors
  detectors.py         leave-one-subject-out detectors across the capacity range
  stats.py             subject-clustered bootstrap, interaction, exact permutation
  data_earndb.py       tone-pip loader and annotation-mapping validation
  data_mendeley.py     click loader
  analysis_earndb.py   agreement, discrimination, interaction, calibration, baselines
  analysis_mendeley.py six-annotator replication and label-variance estimate
  robustness.py        seeds, condition level, exact permutation, influence
  figures.py           publication figures
run_all.py  reproduces every reported number and figure
```

## Evaluation notes

* **Subject-disjoint throughout.** Every fold, every bootstrap resample and
  every permutation operates on whole listeners. Records from one listener are
  not independent, and standardisation, matched-filter templates and calibration
  maps are fitted inside training folds only.
* **Each detector is trained under the label it is scored against**, so the
  comparison reproduces what a study adopting that label would have concluded.
* **Discrimination uses raw statistics.** Per-fold calibration maps are not
  globally monotone and would alter pooled AUROC, so calibration is reported
  separately from discrimination.
* **Hyperparameters are fixed a priori** and are not tuned on these data. The
  detectors are untuned reference implementations; the claims concern
  differences between labels holding the detector fixed.
* **Small-sample guards.** With eight listeners a bootstrap can yield
  implausibly small p-values, so the interaction is additionally tested at
  condition level, under an exact subject-level sign-flip permutation whose
  p-value floor is reported, and with each listener omitted in turn.

## Data

Neither corpus is redistributed. See `data/README.md` for download instructions
and `labels/README.md` for the annotation mapping and the checks applied to it.

## Licence

Code is released under the MIT licence (`LICENSE`). The corpora and the
published annotations remain under their own licences.
