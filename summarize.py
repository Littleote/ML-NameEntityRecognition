import pandas as pd
import sys
from pathlib import Path


def main(stats_file: str, summary_file: str, *config: str):
    stats_path = Path(stats_file)
    with open(stats_path) as handler:
        f1 = handler.readlines()[7].split("\t")[-1].strip(" \n\t\r")
    summary_path = Path(summary_file)
    index = 0
    params = {"F1": f1}
    params = pd.DataFrame({k: [v] for k, v in params.items()})
    while len(config) >= index + 2:
        params[config[index]] = config[index + 1]
        index += 2
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = pd.concat([summary, params])
    else:
        summary = params
    summary.to_csv(summary_file, index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
