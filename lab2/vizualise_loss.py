import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
from pathlib import Path


def find_highest_experiment(directory: str):
    """
    Find the highest numbered experiment in a directory containing subdirectories named exp1, exp2, etc.
    """
    dir_path = Path(directory)
    
    max_exp = -1
    for subdir in dir_path.iterdir():
        if subdir.is_dir() and subdir.name.startswith("exp"):
            try:
                exp_num = int(subdir.name[3:])
                max_exp = max(max_exp, exp_num)
            except ValueError:
                continue
    
    assert max_exp > -1
    return max_exp

if __name__ == "__main__":
    max_exp = find_highest_experiment("runs")
    
    log_dir = f"runs/exp{max_exp}"

    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "events" in f]
    
    event_file = event_files[0]

    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    scalar_keys = event_acc.Tags()["scalars"]
    # print("Available Scalars:", scalar_keys)


    train_loss_events = event_acc.Scalars("train_loss")
    val_loss_events = event_acc.Scalars("val_loss")

    train_loss = [e.value for e in train_loss_events]
    val_loss = [e.value for e in val_loss_events]

    t_loss_epoch = [np.mean(train_loss[i:i+960]) for i in range(0, 7680, 960)]
    v_loss_epoch = [np.mean(val_loss[i:i+240]) for i in range(0, 1920, 240)]

    plt.plot(np.arange(1, 9), t_loss_epoch, c="r", label="training loss")
    plt.plot(np.arange(1, 9), v_loss_epoch, c="g", label="validation loss")
    plt.legend(loc="best")
    plt.title("loss for each epoch")
    plt.show()
