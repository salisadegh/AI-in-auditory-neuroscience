#!/usr/bin/env python3
"""Reproduce every reported result and figure.

Usage:
    python run_all.py

Set ABR_DATA to override the default data directory. See data/README.md for
how to obtain the two corpora; the annotation files in labels/ are included.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import analysis_earndb
import analysis_mendeley
import figures
import robustness


def main():
    print("[1/4] tone-pip corpus: label agreement, discrimination, interaction")
    earndb = analysis_earndb.run()
    print(f"      listeners={earndb['n_listeners']} conditions={earndb['n_conditions']} "
          f"kappa={earndb['agreement']['kappa']:.3f}")

    print("[2/4] click corpus: six-annotator replication and label variance")
    mendeley = analysis_mendeley.run()
    print(f"      inter-annotator kappa median="
          f"{mendeley['raters']['inter_rater_kappa_median']:.3f}")

    print("[3/4] robustness: seeds, condition level, exact permutation, influence")
    guards = robustness.run()
    for key, value in guards["condition_level"].items():
        print(f"      {key}: condition-level p={value['p']:.4f}, "
              f"permutation p={guards['exact_permutation'][key]['p']:.4f}")

    print("[4/4] figures")
    for name in figures.run():
        print("      ", name)
    print("done - results/ and figures/ populated")


if __name__ == "__main__":
    main()
