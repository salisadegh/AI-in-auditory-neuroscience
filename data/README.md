# Obtaining the corpora

Neither corpus is redistributed here. Both are public; download them into this
directory (or set `ABR_DATA` to point elsewhere).

## Tone-pip corpus → `data/earndb/`

Averaged-waveform auditory brainstem responses from eight normal-hearing
listeners at 1 and 4 kHz.

* Source: https://physionet.org/content/earndb/1.0.0/
* Only the `average/` folder is required (~20 MB); the raw single-sweep archive
  is not used.
* Place the WFDB pairs directly in `data/earndb/`, so that files appear as
  `data/earndb/N1_evoked_ave100_F1_R1.hea` and so on.

```bash
mkdir -p data/earndb
wget -r -N -c -np -nd -P data/earndb \
  https://physionet.org/files/earndb/1.0.0/average/
```

## Click corpus → `data/mendeley/`

Click-evoked responses from eight listeners, exported as CSV by a commercial
evoked-potential system.

* Source: https://data.mendeley.com/datasets/4yb9772dff/1
* Only the human recordings are required. Preserve the per-participant folder
  layout, so that files appear as `data/mendeley/P1/TEMP20db.TXT` and so on.

## Annotations

The expert annotations are already included under `labels/`. They were
published as supplementary material to the multicentre study cited in the
manuscript and are redistributed here in machine-readable form under that
article's Creative Commons licence, with the mapping onto corpus records
described in `labels/README.md`.
