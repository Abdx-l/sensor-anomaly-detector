# Sensor Anomaly Detector

Detect spikes, drift, and flatline periods in sensor readings. The CLI combines a rolling z-score with a small isolation forest over value, change, and local variability. It writes a CSV of flagged samples, grouped events, and an SVG plot.

```sh
python detect.py examples/readings.csv --output results
python -m unittest discover -s tests -v
```

The forest is implemented with NumPy and random split trees. `--contamination` sets the fraction of the highest scoring samples flagged by the forest. The example uses synthetic data with injected anomalies so detections can be checked against known positions.

