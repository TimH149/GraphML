"""
Script to convert edge_index.pt file to .txt file
"""

import torch
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
input_dir = project_dir / "Synthetic2" / "graph_edges"
output_dir = project_dir / "data" / "Hyperspheres_10_10_0"

output_dir.mkdir(parents=True, exist_ok=True)

homo_vals = ["homo025", "homo045", "homo065", "homo085"]

for homo_val in homo_vals:
    edge_index = torch.load(input_dir / f"{homo_val}_edge_index.pt")

    # [2, E] -> [E, 2]
    edges = edge_index.T.cpu().numpy()

    output_file = output_dir / f"{homo_val}.txt"

    np.savetxt(output_file, edges, fmt="%d")

    print(f"{output_file} saved")