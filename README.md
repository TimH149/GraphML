# GraphML
First clone the dataset github of Tianqi to get all data: 
git clone https://github.com/Tianqi-py/MLGNC.git


To generate deepwalk embeddings for e.g. blogcatalog use the following command: 
python generate_deepwalk_embeddings.py --edges data/blogcatalog.mat --output data/blogcatalog/blogcatalog.emb