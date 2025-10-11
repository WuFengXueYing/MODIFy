import re
import matplotlib.pyplot as plt
from datetime import datetime


with open("../../result/b52b1a7d/running.log", "r") as f:
    lines = f.readlines()


timestamp_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
session_start_pattern = r"Model Parameters: hash_id=b52b1a7d"
epoch_pattern = r"Epoch (\d+)/\d+, training loss: ([\d.]+)"


sessions = []
current_session = {"start_time": None, "epochs": []}

for line in lines:

    timestamp_match = re.search(timestamp_pattern, line)
    if not timestamp_match:
        continue
    timestamp_str = timestamp_match.group(1)
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")


    if re.search(session_start_pattern, line):

        if current_session["start_time"]:
            sessions.append(current_session)

        current_session = {"start_time": timestamp, "epochs": []}


    epoch_match = re.search(epoch_pattern, line)
    if epoch_match and current_session["start_time"]:
        epoch_num = int(epoch_match.group(1))
        loss = float(epoch_match.group(2))
        current_session["epochs"].append((epoch_num, loss))


if current_session["start_time"]:
    sessions.append(current_session)


latest_session = max(sessions, key=lambda s: s["start_time"])
epochs, losses = zip(*latest_session["epochs"])




plt.plot(epochs, losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.grid(True)

plt.savefig("../Fig/sn_convergence_curve.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()

plt.close()