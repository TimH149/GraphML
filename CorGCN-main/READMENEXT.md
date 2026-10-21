# Running CorGCN using provided dataset

## Dataset
We assume that the data is downloaded from:
`https://github.com/Tianqi-py/MLGNC/tree/main/Graph%20Generator%20Model`

## How to run
After setting up the environment as explained in this folder's `README.md`, you can run the CorGCN model on the dataset described above by following the steps below.

1. First navigate to CorGCN folder
2. Make sure best_param folder exists inside CorGCN folder.
3. Run CorGCN on real-world dataset (e.g. PCG) using:
    
    `python run_CorGCN.py --model_name corgcn --data_name mf_pcg --mf_root ../data/ --split_names split_0.pt split_1.pt split_2.pt --use_gpu --gpu_id 0 --lr 0.001 --bs 1024 --k_num 5`
4. Run CorGCN on synthetic dataset (e.g. Synthetic025) using:
    
    `python run_CorGCN.py --model_name corgcn --data_name mf_hyper --mf_root ../data/ --homo_level homo025 --split_names split_0.pt split_1.pt split_2.pt --use_gpu --gpu_id 0 --lr 0.001 --bs 1024 --k_num 5`