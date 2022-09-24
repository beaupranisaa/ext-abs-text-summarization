# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console
from IPython.display import clear_output
import numpy as np
import os
import re

console = Console(record=True)

def display_dataset(ds):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(ds.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)
    
def checker(model_params):
    pass

def get_last_checkpoint(path):
    if not os.path.exists(path):
        raise ValueError("No checkpoints to resume, please start training to create checkpoints...")
    else:
        content = os.listdir(path)
        if len(content) != 9:
            raise ValueError("No checkpoints to resume, please start training to create checkpoints...")
        last_checkpoint = int(np.load(f"{path}/current_epoch.npy"))
        print(f"[Resuming....] at EPOCH {last_checkpoint}")
        return last_checkpoint + 1      
