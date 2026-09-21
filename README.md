# MODIFy

MODIFy: A Multi-Modal Anomaly Diagnosis Framework with Diffusion-Enhanced Adaptive Fusion in Microservices

## Overview

MODIFy is a graph-based multi-modal anomaly diagnosis framework for microservice systems. It combines heterogeneous signals from logs, metrics, traces, and service dependency graphs to identify abnormal services and localize root causes.

The project is organized around a training pipeline that:

- preprocesses raw microservice datasets;
- aligns heterogeneous data into time windows;
- builds graph-structured chunks for training/testing;
- trains a multi-source fusion model;
- evaluates diagnosis and localization performance.

This repository contains the core model implementation, preprocessing pipeline, and experiment runner.

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

### 1. Model implementation

- `codes/model.py`
  - defines the graph model, transformer encoder, diffusion-based trace module, CGLU fusion, and main model.

- `codes/base.py`
  - contains training, validation, early stopping, model saving, and evaluation logic.

- `codes/main.py`
  - entry point for experiments.

### 2. Data preprocessing

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

Common options:

```bash
python align.py --name SN --chunk_lenth 6 --test_ratio 0.3 --threshold 1
```

### Parameters in `align.py`

- `--name`: dataset name (`SN` or `TT`)
- `--chunk_lenth`: sliding window length
- `--test_ratio`: proportion of test data
- `--threshold`: minimum overlap required to label a window as faulty
- `--delete`: remove generated chunks while keeping preprocessed data
- `--delete_all`: delete all generated chunk files under the target dataset
- `--concat`: merge chunk files from multiple runs

After preprocessing, the following files are typically generated under `codes/chunks/<name>/`:

```text
metadata.json
chunk_train.pkl
chunk_test.pkl
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

### Example training command

```bash
cd codes
python main.py \
    --data SN \
    --gpu true \
    --epoches 70 \
    --batch_size 256 \
    --lr 0.001 \
    --patience 10 \
    --use_transformer true \
    --use_CGLU true \
    --use_TraceDifussion true
```

## Important Training Arguments

`main.py` supports the following major arguments:

- `--random_seed`
- `--gpu`
- `--epoches`
- `--batch_size`
- `--lr`
- `--patience`
- `--data`
- `--seq_len` (recommended to add if missing)
- `--result_dir`
- `--self_attn`
- `--fuse_dim`
- `--alpha`
- `--beta`
- `--locate_hiddens`
- `--detect_hiddens`
- `--detector_rank`
- `--locator_rank`
- `--log_dim`
- `--trace_hiddens`
- `--metric_hiddens`
- `--graph_hiddens`
- `--attn_head`
- `--use_transformer`
- `--use_CGLU`
- `--use_TraceDifussion`

## Notes on the Current Code

The repository is designed for research and experimentation. There are a few implementation details to be aware of:

1. The training pipeline expects processed chunks to exist before running `main.py`.
2. Some data paths are relative to the working directory, so it is important to run scripts from the correct folders.
3. `main.py` references `params["seq_len"]`; in the current code, this value is set only if `--seq_len` is provided or if a chunk length is assigned. If you run it directly, you may need to add the argument explicitly.
4. The project uses a custom time-based split for sliding windows and includes logic to avoid raw timestamp leakage between train and test sets.

## Output

Training results are written to the `result/` directory.

Example:

```text
result/
└── <hash_id>/
    ├── params.json
    ├── running.log
    └── model.ckpt
```

The script also appends summary information to `result/experiments.txt`.

## Evaluation Metrics

The project evaluates both anomaly detection and root-cause localization using metrics such as:

- Precision
- Recall
- F1
- HR@1
- HR@3
- HR@5
- nDCG@1
- nDCG@3
- nDCG@5

## Typical Run Sequence

```bash
# 1. install packages
pip install -r requirements.txt

# 2. prepare raw dataset under ./datasets

# 3. preprocess data
cd codes/preprocess
python parse_raw_logs.py
python parse_raw_metrics.py
python parse_raw_traces.py
python parse_raw_records.py
python align.py --name SN

# 4. train the model
cd ..
python main.py --data SN --gpu true
```

## Known Practical Considerations

- The model is computationally heavy and is intended for research use.
- GPU support is recommended for training efficiency.
- Some scripts rely on specific folder names and file layouts from the benchmark datasets.
- If you are running in a fresh environment, ensure all dependencies are installed before data preprocessing.

## License

Please check the repository license file if one is present. If no explicit license file is included, treat the source as research code and use it under the repository owner’s terms.

## Citation / Reference

If this project is used in academic work, cite the corresponding paper or project reference associated with the MODIFy framework.

## Acknowledgements

This project draws on research in:

- graph neural networks;
- multi-modal anomaly detection;
- diffusion-based representation learning;
- microservice fault diagnosis;
- transformer-based temporal modeling.

---

This README is designed to match the actual project structure and execution flow present in the codebase.
