# align.py: process & align multi-source data (logs/traces/metrics) into chunks,
# then split into train/test sets.
#
# IMPORTANT (fix temporal leakage):
# - We split on the *raw time axis* first (per record run), then only keep windows
#   fully on one side. No interval in train shares any raw timestamp with any
#   interval in test.

import string
import random
import os
import pickle
from collections import defaultdict

import numpy as np

from single_process import deal_logs, deal_traces, deal_metrics
from util import *

chunkids = set()
src = string.ascii_letters + string.digits


def get_chunkid():
    """Generate a unique (random) chunk id."""
    while True:
        chunkid = random.sample(src, 8)
        random.shuffle(chunkid)
        chunkid = ''.join(chunkid)
        if chunkid not in chunkids:
            chunkids.add(chunkid)
            return chunkid


def _compute_split_time(start: int, end: int, chunk_lenth: int, test_ratio: float) -> int:
    """Compute a time-based split point (per run) and clamp to keep both sides non-empty."""
    assert 0.0 < test_ratio < 1.0
    duration = end - start + 1
    # initial: split by time length
    train_duration = int((1.0 - test_ratio) * duration)
    split_time = start + train_duration  # train uses [start, split_time-1], test uses [split_time, end]

    # Ensure there is at least one full window on each side
    # Train needs: exists s such that s + chunk_lenth - 1 <= split_time - 1  => split_time >= start + chunk_lenth
    min_split = start + chunk_lenth
    # Test needs: exists s such that s + chunk_lenth - 1 <= end and s >= split_time => split_time <= end - chunk_lenth + 1
    max_split = end - chunk_lenth + 1

    split_time = max(split_time, min_split)
    split_time = min(split_time, max_split)
    return split_time


def _build_intervals_inclusive(t_start: int, t_end_inclusive: int, chunk_lenth: int):
    """Sliding windows [(s, s+L-1)] fully inside [t_start, t_end_inclusive]."""
    last_s = t_end_inclusive - chunk_lenth + 1
    if last_s < t_start:
        return []
    return [(s, s + chunk_lenth - 1) for s in range(t_start, last_s + 1)]


# def _annotate_intervals(intervals, faults, threshold: int):
#     """Assign labels per interval based on fault overlap."""
#     labels = [-1] * len(intervals)
#     for chunk_idx, (s, e) in enumerate(intervals):
#         for (fs, fe, culprit) in faults:
#             overlap = 0
#             # overlap length between [s,e] and [fs,fe]
#             if s >= fs and s <= fe:
#                 overlap = fe - s + 1
#             elif e >= fs and e <= fe:
#                 overlap = e - fs + 1
#             elif fs >= s and fs <= e:
#                 overlap = e - fs + 1
#             elif fe >= s and fe <= e:
#                 overlap = fe - s + 1
#
#             if overlap >= threshold:
#                 labels[chunk_idx] = culprit
#             if overlap > 0:
#                 break
#     return labels
def _annotate_intervals(intervals, faults, threshold: int):
    labels = [-1] * len(intervals)
    for chunk_idx, (s, e) in enumerate(intervals):
        for (fs, fe, culprit) in faults:
            left = max(s, fs)
            right = min(e, fe)
            overlap = max(0, right - left + 1)  # closed interval

            if overlap >= threshold:
                labels[chunk_idx] = culprit
                break  # 只在满足阈值时才停止找
    return labels

# Generate intervals and labels, with time-based split (per records idx)
def get_basic(info, idx, name, chunk_lenth=10, threshold=1, test_ratio=0.3, **kwargs):
    records = read_json(os.path.join("./parsed_data", name, "records" + idx + ".json"))

    faults = [
        (f_record["s"], f_record["e"], info.service2nid[f_record["service"]])
        for f_record in records["faults"]
    ]

    start, end = records["start"], records["end"]

    # --- time-based split (per run) ---
    split_time = _compute_split_time(start, end, chunk_lenth, test_ratio)

    # Train uses ONLY timestamps < split_time
    train_intervals = _build_intervals_inclusive(start, split_time - 1, chunk_lenth)
    # Test uses ONLY timestamps >= split_time
    test_intervals = _build_intervals_inclusive(split_time, end, chunk_lenth)

    intervals = train_intervals + test_intervals
    split_flags = ["train"] * len(train_intervals) + ["test"] * len(test_intervals)

    labels = _annotate_intervals(intervals, faults, threshold)

    if len(intervals) == 0:
        raise ValueError(
            f"No valid intervals built for records{idx}.json. "
            f"Check start/end ({start}-{end}) and chunk_lenth={chunk_lenth}."
        )

    print(
        "# run {rid}: start={s}, end={e}, split_time={sp} (train<{sp}, test>={sp}), "
        "intervals: train={tn}, test={vn}, total={tot}".format(
            rid=idx, s=start, e=end, sp=split_time,
            tn=len(train_intervals), vn=len(test_intervals), tot=len(intervals)
        )
    )
    if len(intervals) > 0:
        print('# intervals span: first=[{},{}], last=[{},{}]'.format(
            intervals[0][0], intervals[0][1], intervals[-1][0], intervals[-1][1]
        ))

    return intervals, labels, split_flags, split_time


def _load_or_compute_traces(aim_dir, intervals, info, idx, name, chunk_lenth):
    p = os.path.join(aim_dir, "traces.pkl")
    if os.path.exists(p):
        with open(p, "rb") as fr:
            traces = pickle.load(fr)
        # validate
        if "latency" in traces and traces["latency"].shape[0] == len(intervals):
            return traces
    traces = deal_traces(intervals, info, idx, name=name, chunk_lenth=chunk_lenth)
    with open(p, "wb") as fw:
        pickle.dump(traces, fw)
    return traces


def _load_or_compute_metrics(aim_dir, intervals, info, idx, name, chunk_lenth):
    p = os.path.join(aim_dir, "metrics.pkl")
    if os.path.exists(p):
        with open(p, "rb") as fr:
            metrics = pickle.load(fr)
        if isinstance(metrics, np.ndarray) and metrics.shape[0] == len(intervals):
            return metrics
    metrics = deal_metrics(intervals, info, idx, name=name, chunk_lenth=chunk_lenth)
    with open(p, "wb") as fw:
        pickle.dump(metrics, fw)
    return metrics


def _load_or_compute_logs(aim_dir, intervals, info, idx, name):
    p = os.path.join(aim_dir, "logs.pkl")
    if os.path.exists(p):
        with open(p, "rb") as fr:
            logs = pickle.load(fr)
        if isinstance(logs, np.ndarray) and logs.shape[0] == len(intervals):
            return logs
    logs = deal_logs(intervals, info, idx, name=name)
    with open(p, "wb") as fw:
        pickle.dump(logs, fw)
    return logs


def get_chunks(info, idx, name, chunk_lenth=10, test_ratio=0.3, **kwargs):
    intervals, labels, split_flags, split_time = get_basic(
        info, idx, name=name, chunk_lenth=chunk_lenth, test_ratio=test_ratio, **kwargs
    )

    aim_dir = os.path.join("../chunks", name, idx)
    os.makedirs(aim_dir, exist_ok=True)

    traces = _load_or_compute_traces(aim_dir, intervals, info, idx, name, chunk_lenth)
    metrics = _load_or_compute_metrics(aim_dir, intervals, info, idx, name, chunk_lenth)
    logs = _load_or_compute_logs(aim_dir, intervals, info, idx, name)

    print("*** Aligning multi-source data...")
    chunks = defaultdict(dict)

    # IMPORTANT: don't shadow the function argument `idx` (run id)
    for interval_i in range(len(intervals)):
        chunk_id = get_chunkid()
        chunks[chunk_id]["traces"] = traces["latency"][interval_i]  # [node_num, chunk_lenth, 2]
        chunks[chunk_id]["metrics"] = metrics[interval_i]
        chunks[chunk_id]["logs"] = logs[interval_i]
        chunks[chunk_id]["culprit"] = labels[interval_i]

        # --- extra metadata for safe splitting / debugging ---
        chunks[chunk_id]["split"] = split_flags[interval_i]  # "train" or "test"
        chunks[chunk_id]["interval"] = intervals[interval_i]  # (s, e)
        chunks[chunk_id]["run_id"] = int(idx)
        chunks[chunk_id]["split_time"] = split_time

    return chunks


def get_all_chunks(name, chunk_lenth=10, test_ratio=0.3, **kwargs):
    aim_dir = os.path.join("../chunks", name)
    os.makedirs(aim_dir, exist_ok=True)

    # bench = "TrainTicket" if name == "TT" else "SocialNetwork"
    bench = "TrainTicket" if name == "TT" else "SocialNetwork"
    info = Info(bench)
    print('# Node num:', info.node_num)

    chunks = {}
    idx = 0
    while True:
        rec_path = os.path.join("./parsed_data", name, "records" + str(idx) + ".json")
        if os.path.exists(rec_path):
            print("\n\n", "^" * 20, "Now dealing with", idx, "^" * 20)
            new_chunks = get_chunks(
                info, str(idx), chunk_lenth=chunk_lenth, test_ratio=test_ratio, name=name, **kwargs
            )
            chunks.update(new_chunks)
            idx += 1
        else:
            break

    print("# Data Genenaration Batch Size: ", idx)
    with open(os.path.join(aim_dir, "chunks.pkl"), "wb") as fw:
        pickle.dump(chunks, fw)

    # Update metadata
    info.add_info("chunk_lenth", chunk_lenth)
    info.add_info("chunk_num", len(chunks))
    info.add_info("edges", info.edges)
    info.add_info("event_num", chunks[list(chunks.keys())[0]]["logs"].shape[-1])

    meta_path = os.path.join(aim_dir, "metadata.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)
    json_pretty_dump(info.metadata, meta_path)

    return chunks


def split_chunks(name, test_ratio=0.3, concat=False, **kwargs):
    """Split chunks into train/test.

    New behavior (no leakage): if chunks contain the key `split`, we trust it and
    split deterministically by time. Otherwise, we fall back to the old random split.
    """
    chunks = {}

    if concat:
        print("*** Concating chunks...")
        chunk_num = 0
        metadata = None
        for dir in os.listdir("../chunks"):
            if not dir.startswith(name[0]):
                continue
            if os.path.exists(os.path.join("../chunks", dir, "chunks.pkl")):
                with open(os.path.join("../chunks", dir, "chunks.pkl"), "rb") as fr:
                    chunks.update(pickle.load(fr))
                metadata = read_json(os.path.join("../chunks", dir, "metadata.json"))
                chunk_num += metadata["chunk_num"]

        os.makedirs(os.path.join("../chunks", name[0]), exist_ok=True)
        if metadata is not None:
            metadata["chunk_num"] = chunk_num
            json_pretty_dump(metadata, os.path.join("../chunks", name[0], "metadata.json"))

    elif os.path.exists(os.path.join("../chunks", name, "chunks.pkl")):
        with open(os.path.join("../chunks", name, "chunks.pkl"), "rb") as fr:
            chunks.update(pickle.load(fr))
    else:
        # If need to generate, pass test_ratio so get_basic can compute split_time.
        chunks = get_all_chunks(name=name, test_ratio=test_ratio, **kwargs)

    print("\n *** Spliting chunks into training and testing sets...")

    # --- preferred: time-based split already stored per chunk ---
    has_split_flag = len(chunks) > 0 and all(isinstance(v, dict) and ("split" in v) for v in chunks.values())

    if has_split_flag:

        train_chunks = {k: v for k, v in chunks.items() if v.get("split") == "train"}
        test_chunks = {k: v for k, v in chunks.items() if v.get("split") == "test"}
        dropped = len(chunks) - len(train_chunks) - len(test_chunks)
        if dropped > 0:
            print(f"[WARN] {dropped} chunks have invalid split flag and were ignored.")

        train_num, test_num, chunk_num = len(train_chunks), len(test_chunks), len(train_chunks) + len(test_chunks)
        print(f"# time-based split (no leakage): train={train_num}, test={test_num}, total_used={chunk_num}")

    else:
        # --- fallback: old random split (may leak for sliding windows) ---
        print("[WARN] chunks do not contain 'split' metadata. Falling back to RANDOM split (may leak).")
        chunk_num = len(chunks)
        chunk_hashids = np.array(list(chunks.keys()))
        chunk_idx = list(range(chunk_num))

        train_num = int((1 - test_ratio) * chunk_num)
        test_num = int(test_ratio * chunk_num)
        np.random.shuffle(chunk_idx)

        train_idx = chunk_idx[:train_num]
        test_idx = chunk_idx[train_num:train_num + test_num]

        train_chunks = {k: chunks[k] for k in chunk_hashids[train_idx]}
        test_chunks = {k: chunks[k] for k in chunk_hashids[test_idx]}

    aim = name[0] if concat else name
    with open(os.path.join("../chunks", aim, "chunk_train.pkl"), "wb") as fw:
        pickle.dump(train_chunks, fw)
    with open(os.path.join("../chunks", aim, "chunk_test.pkl"), "wb") as fw:
        pickle.dump(test_chunks, fw)

    # --- Print statistics ---
    label_count = {}
    for _, v in chunks.items():
        label = v.get('culprit', -1)
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1

    train_labels = [v.get('culprit', -1) != -1 for _, v in train_chunks.items()]
    test_labels = [v.get('culprit', -1) != -1 for _, v in test_chunks.items()]

    if len(train_labels) > 0:
        print("# train chunks: {}/{} ({:.4f}%)".format(sum(train_labels), len(train_labels), 100 * (sum(train_labels) / len(train_labels))))
    if len(test_labels) > 0:
        print("# test chunks: {}/{} ({:.4f}%)".format(sum(test_labels), len(test_labels), 100 * (sum(test_labels) / len(test_labels))))

    for label in sorted(list(label_count.keys())):
        if label > -1:
            print('Node {} have {} faulty chunks'.format(label, label_count[label]))


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--concat", action="store_true")
parser.add_argument("--delete_all", action="store_true")
parser.add_argument("--delete", action="store_true", help="just remove the final chunks and retain pre-processed data")
parser.add_argument("--threshold", default=1, type=int)
parser.add_argument("--chunk_lenth", default=6, type=int)
parser.add_argument("--test_ratio", default=0.3, type=float)
parser.add_argument("--name", required=True, help="The system name")
params = vars(parser.parse_args())


if "__main__" == __name__:
    aim_dir = os.path.join("../chunks", params['name'])

    if params['delete_all']:
        _input = input("Do you really want to delete all previous files?! Input yes if you are so confident.\n")
        flag = (_input.lower() == 'yes')
        if flag and os.path.exists(aim_dir) and len(aim_dir) > 2:
            import shutil
            shutil.rmtree(aim_dir)
        else:
            print("Thank you for thinking twice!")
            exit()

    if params['delete'] and os.path.exists(os.path.join(aim_dir, "chunks.pkl")):
        os.remove(os.path.join(aim_dir, "chunks.pkl"))

    split_chunks(**params)
