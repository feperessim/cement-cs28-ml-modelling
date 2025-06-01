import numpy as np


def print_scores(scores, METRICS, METRICS_DICT):
    for phase in ["train", "test"]:
        print("******")
        print(f"[{phase.upper()}]")
        print("******")
        for metric in METRICS:
            name = METRICS_DICT[metric]
            print(
                f"{name}: %.3f (%.3f)"
                % (
                    np.mean(scores[f"{phase}_" + metric]),
                    np.std(scores[f"{phase}_" + metric]),
                )
            )
        print("\n======================\n")
