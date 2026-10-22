"""Rolling z-score and small isolation forest for time-series anomalies."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import numpy as np


def rolling_features(values: np.ndarray, window: int = 20) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 1 or len(values) < window or window < 3 or not np.all(np.isfinite(values)):
        raise ValueError("Expected finite one-dimensional samples longer than the window")
    baseline = np.empty(len(values))
    spread = np.empty(len(values))
    for i in range(len(values)):
        history = values[max(0,i-window):i]
        if len(history) < 3:
            history = values[:window]
        baseline[i] = np.median(history)
        spread[i] = max(1e-6, 1.4826*np.median(np.abs(history-np.median(history))))
    z = np.abs(values-baseline)/spread
    delta = np.diff(values, prepend=values[0])
    variance = np.array([np.std(values[max(0,i-window+1):i+1]) for i in range(len(values))])
    features = np.column_stack([values, delta, variance])
    return z, features


def average_path(n: int) -> float:
    return 0 if n <= 1 else 2*(sum(1/i for i in range(1,n)))-(2*(n-1)/n)


class Tree:
    def __init__(self, data: np.ndarray, rng: np.random.Generator, depth: int, maximum: int):
        self.size = len(data)
        self.axis = None
        self.left = self.right = None
        if self.size <= 1 or depth >= maximum:
            return
        candidates = [i for i in range(data.shape[1]) if np.ptp(data[:,i]) > 0]
        if not candidates:
            return
        self.axis = int(rng.choice(candidates))
        low, high = np.min(data[:,self.axis]), np.max(data[:,self.axis])
        self.split = rng.uniform(low,high)
        mask = data[:,self.axis] < self.split
        if not np.any(mask) or np.all(mask):
            self.axis = None
            return
        self.left = Tree(data[mask],rng,depth+1,maximum)
        self.right = Tree(data[~mask],rng,depth+1,maximum)

    def path(self, row: np.ndarray, depth: int = 0) -> float:
        if self.axis is None:
            return depth + average_path(self.size)
        return (self.left if row[self.axis] < self.split else self.right).path(row,depth+1)


def forest_scores(features: np.ndarray, trees: int = 64, sample_size: int = 64, seed: int = 7) -> np.ndarray:
    if trees < 1 or sample_size < 2:
        raise ValueError("Invalid forest parameters")
    rng = np.random.default_rng(seed)
    count = min(sample_size,len(features))
    models = [Tree(features[rng.choice(len(features),count,replace=False)],rng,0,math.ceil(math.log2(count))) for _ in range(trees)]
    paths = np.array([sum(tree.path(row) for tree in models)/trees for row in features])
    return 2 ** (-paths / max(average_path(count),1e-9))


def detect(values: np.ndarray, window: int = 20, contamination: float = .04, seed: int = 7) -> dict:
    if not 0 < contamination < .5:
        raise ValueError("contamination must be between 0 and 0.5")
    z, features = rolling_features(values,window)
    scores = forest_scores(features,seed=seed)
    threshold = np.quantile(scores,1-contamination)
    flat = np.zeros(len(values),dtype=bool)
    for i in range(window-1,len(values)):
        flat[i] = np.ptp(values[i-window+1:i+1]) < 1e-6
    flags = (z > 5) | (scores >= threshold) | flat
    events = []
    start = None
    for i, flag in enumerate(np.append(flags,False)):
        if flag and start is None: start = i
        if not flag and start is not None:
            events.append({"start":start,"end":i-1,"peak_score":round(float(np.max(scores[start:i])),4)})
            start = None
    return {"z":z,"scores":scores,"flags":flags,"events":events,"flat":flat}


def svg(values: np.ndarray, result: dict) -> str:
    width,height,pad = 900,320,30
    low, high = float(np.min(values)),float(np.max(values))
    scale = max(high-low,1e-9)
    point = lambda i: (pad + i*(width-2*pad)/max(1,len(values)-1), height-pad-(values[i]-low)*(height-2*pad)/scale)
    line = " ".join(f"{x:.1f},{y:.1f}" for x,y in (point(i) for i in range(len(values))))
    dots = "".join(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3' fill='#c33131'/>" for x,y in (point(i) for i in np.flatnonzero(result["flags"])))
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' role='img' aria-label='Sensor readings with flagged anomalies'><rect width='100%' height='100%' fill='white'/><polyline points='{line}' fill='none' stroke='#2265ad' stroke-width='1.5'/>{dots}</svg>"


def load_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="",encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any("value" not in row for row in rows):
        raise ValueError("CSV needs a value column")
    return [row.get("timestamp",str(i)) for i,row in enumerate(rows)], np.array([float(row["value"]) for row in rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--window",type=int,default=20)
    parser.add_argument("--contamination",type=float,default=.04)
    args = parser.parse_args()
    timestamps, values = load_csv(args.csv)
    result = detect(values,args.window,args.contamination)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"flags.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp","value","z_score","forest_score","flagged"])
        writer.writerows((stamp,float(value),round(float(z),3),round(float(score),3),bool(flag)) for stamp,value,z,score,flag in zip(timestamps,values,result["z"],result["scores"],result["flags"]))
    (args.output/"events.json").write_text(json.dumps(result["events"],indent=2),encoding="utf-8")
    (args.output/"signal.svg").write_text(svg(values,result),encoding="utf-8")
    print(f"Flagged {int(np.sum(result['flags']))} samples in {len(result['events'])} event(s)")

