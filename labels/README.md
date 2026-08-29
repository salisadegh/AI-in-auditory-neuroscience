# Expert annotations

Machine-readable mappings of published audiologist annotations onto the records
of the two corpora. These files are the reusable artefact of this repository:
neither corpus distributes annotations, and the published tables give them in a
form that must be aligned to record identifiers before use.

## Files

| File | Key | Value |
|---|---|---|
| `earndb_expert_labels.json` | `"<subject>\|<carrier frequency kHz>\|<level dB>"` | `0` absent, `1` present |
| `mendeley_consensus_labels.json` | `"<subject>\|<level dB>"` | majority vote of six annotators |
| `mendeley_rater_labels.json` | `"<annotator>"` → `"<subject>\|<level dB>"` | per-annotator judgement |

The tone-pip annotations come from a single audiologist and cover 25–100 dB in
5 dB steps (204 conditions). The click annotations come from six independent
audiologists; 86 conditions are annotated by all six.

## Alignment

The published tone-pip table heads its level column with a sensation-level unit
while the corpus records levels in peak-equivalent SPL. The identity mapping was
adopted after testing alternatives, and `data_earndb.validate_label_mapping()`
reproduces those checks:

* every annotated (subject, frequency, level) triple matches an existing record,
  and no annotation lacks a record — the direction that would indicate a broken
  mapping;
* shifting the annotation levels by ±5 or ±10 dB matches strictly fewer records
  than the identity mapping;
* over 25–100 dB the populated-cell pattern of the table agrees with the corpus
  record inventory in 92.6% of cells, the remainder being records present in the
  corpus but absent from the table.

This alignment remains an inference. Direct confirmation from the annotating
authors would remove the residual risk, and the manuscript states this.
