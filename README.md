# MODIFy

MODIFy: A Multi-Modal Anomaly Diagnosis Framework with Diffusion-Enhanced Adaptive Fusion in Microservices

## Overview

MODIFy is a graph-based multi-modal anomaly diagnosis framework for microservice systems. It combines heterogeneous signals from logs, metrics, traces, and service dependency graphs to identify abnormal services and localize root causes.

## Repository Structure

```text
MODIFy/
├── README.md
├── requirements.txt
├── codes/
│   ├── main.py
│   ├── model.py
│   ├── base.py
│   ├── utils.py
│   ├── experimental result.ipynb
│   └── preprocess/
│       ├── align.py
│       ├── drain3.ini
│       ├── parse_raw_logs.py
│       ├── parse_raw_metrics.py
│       ├── parse_raw_records.py
│       ├── parse_raw_traces.py
│       ├── single_process.py
│       ├── util.py
│       └── __pycache__/
└── datasets/
    ├── SN Dataset/
    └── TT Dataset/
```

## Main Components
### Data preprocessing

- `codes/preprocess/parse_raw_logs.py`
  - extracts log templates using Drain3.

- `codes/preprocess/parse_raw_metrics.py`
  - processes metric CSV files.

- `codes/preprocess/parse_raw_traces.py`
  - converts raw trace spans into structured trace data.

- `codes/preprocess/parse_raw_records.py`
  - processes record/fault metadata.

- `codes/preprocess/align.py`
  - aligns logs, traces, and metrics into chunks, then splits into train/test sets.

## Environment Setup

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

The project depends on:

- PyTorch
- DGL
- NumPy
- Pandas
- scikit-learn
- tick
- transformers
- drain3
- matplotlib

Recommended Python version: 3.8 or 3.9.

## Dataset Preparation

This project expects raw microservice datasets under the `datasets/` directory.

The preprocessing scripts are written to read data from paths like:

```text
datasets/
├── SN Dataset/
│   ├── no fault/
│   └── data/
└── TT Dataset/
    ├── no fault/
    └── data/
```

The repository does not include the raw dataset files by default, so you must prepare them separately before running the pipeline.

## Preprocessing Pipeline

Run the preprocessing scripts from the `codes/preprocess` directory:

```bash
cd codes/preprocess
python parse_raw_logs.py
python parse_raw_metrics.py
python parse_raw_traces.py
python parse_raw_records.py
```

Then generate chunked train/test data:

```bash
python align.py --name SN
```

or

```bash
python align.py --name TT
```

## Training

From the `codes/` directory, start training with a dataset:

```bash
cd codes
python main.py --data SN
```

or

```bash
cd codes
python main.py --data TT
```

## Citation / Reference

If this project is used in academic work, cite the corresponding paper or project reference associated with the MODIFy framework.
Zhang, Wujian, et al. "MODIFy: a multi-modal anomaly diagnosis framework with diffusion-enhanced adaptive fusion in microservices." Journal of Systems and Software (2026): 112844.

