"""Generate DeepWalk/Word2Vec embeddings from an edge list or MATLAB graph.

Node IDs in the output are always 0..N-1, matching row order expected by
Graph-MultiFix loaders. For an edge list, IDs must already be row indices or
use --node-ids with one graph ID per line in feature/label row order. For a
MAT file, use --mat-key to select an adjacency matrix; matrix row indices are
used as node IDs.

Defaults follow the provided DBLP example: 10 walks per node, walk length 10,
window 5, 64 dimensions, Skip-gram, hierarchical softmax, 5 epochs, and one
worker. Walks are streamed into Gensim rather than retained as a large list.
"""
import argparse
from contextlib import contextmanager
from pathlib import Path
import random
import zipfile

import numpy as np
from scipy import io as scipy_io
from scipy.sparse import csr_matrix, issparse
from gensim.models import Word2Vec


@contextmanager
def open_edge_data(path: Path):
    """Open a plain edge file or a zip containing one edge-list file."""
    if path.suffix.lower() != ".zip":
        with path.open("rb") as stream:
            yield stream
        return

    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            preferred = [name for name in members
                         if Path(name).name.lower() in {"edges.txt", "edge_list.txt"}]
            if len(preferred) == 1:
                member = preferred[0]
            else:
                raise ValueError(
                    "Zip must contain one edge-list file; extract the desired "
                    "edge-list member and pass that file instead."
                )
        else:
            member = members[0]
        with archive.open(member) as stream:
            yield stream


def detect_delimiter(stream):
    """Infer comma-separated versus whitespace-separated edge lists."""
    position = stream.tell()
    for line in stream:
        stripped = line.strip()
        if stripped and not stripped.startswith(b"#"):
            stream.seek(position)
            return "," if b"," in stripped else None
    stream.seek(position)
    raise ValueError("The edge file is empty")


def read_graph(path: Path, delimiter_arg: str, skip_rows: int, mat_key: str):
    """Return edge pairs and the matrix node count when reading a MAT file."""
    if path.suffix.lower() == ".mat":
        mat = scipy_io.loadmat(str(path))
        if mat_key not in mat:
            available = [key for key in mat if not key.startswith("__")]
            raise KeyError(f"MAT key {mat_key!r} not found. Available keys: {available}")
        matrix = mat[mat_key]
        if issparse(matrix):
            matrix = matrix.tocoo()
            edges = np.column_stack((matrix.row, matrix.col))
            num_nodes = matrix.shape[0]
        else:
            matrix = np.asarray(matrix)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"MAT key {mat_key!r} must be a square adjacency matrix")
            rows, cols = np.nonzero(matrix)
            edges = np.column_stack((rows, cols))
            num_nodes = matrix.shape[0]
        return edges.astype(np.int64, copy=False), int(num_nodes)

    with open_edge_data(path) as stream:
        delimiter = {
            "comma": ",",
            "whitespace": None,
            "tab": "\t",
        }.get(delimiter_arg)
        if delimiter_arg == "auto":
            delimiter = detect_delimiter(stream)
        edges = np.loadtxt(
            stream,
            dtype=np.float64,
            delimiter=delimiter,
            usecols=(0, 1),
            skiprows=skip_rows,
            comments="#",
            ndmin=2,
        )

    if not np.isfinite(edges).all():
        raise ValueError("The edge list contains a non-finite endpoint")
    rounded = np.rint(edges)
    if not np.array_equal(edges, rounded):
        raise ValueError("Edge endpoints must be integer-valued node IDs")
    return rounded.astype(np.int64, copy=False), None


def map_to_row_indices(edges: np.ndarray, node_ids_path: Path | None,
                       num_nodes_arg: int | None):
    if node_ids_path is not None:
        node_ids = np.loadtxt(node_ids_path, dtype=np.int64, ndmin=1)
        if np.unique(node_ids).size != node_ids.size:
            raise ValueError("--node-ids contains duplicate IDs")

        sort_order = np.argsort(node_ids)
        sorted_ids = node_ids[sort_order]
        positions = np.searchsorted(sorted_ids, edges)
        valid = positions < sorted_ids.size
        valid[valid] &= sorted_ids[positions[valid]] == edges[valid]
        if not np.all(valid):
            missing = np.unique(edges[~valid])[:10]
            raise ValueError(
                "Some edge IDs are missing from --node-ids, e.g. "
                + ", ".join(map(str, missing.tolist()))
            )
        return sort_order[positions], int(node_ids.size)

    if edges.size == 0:
        raise ValueError("No edges were read")
    if edges.min() < 0:
        raise ValueError("Node IDs must be non-negative")

    num_nodes = int(num_nodes_arg) if num_nodes_arg is not None else int(edges.max()) + 1
    if edges.max() >= num_nodes:
        raise ValueError(
            f"Edge ID {int(edges.max())} is outside [0, {num_nodes - 1}]. "
            "Use --node-ids for non-contiguous or externally numbered nodes."
        )
    return edges, num_nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", required=True, type=Path,
                        help="Edge list, zip file, or MAT file containing an adjacency matrix")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output Word2Vec text file, e.g. data/blogcatalog/blogcatalog.emb")
    parser.add_argument("--mat-key", default="network",
                        help="Adjacency matrix key in a MAT file (default: network)")
    parser.add_argument("--num-nodes", type=int,
                        help="Total node count, including isolates, for 0-based edge IDs")
    parser.add_argument("--node-ids", type=Path,
                        help="One graph ID per line, in feature/label row order")
    parser.add_argument("--delimiter", choices=("auto", "comma", "whitespace", "tab"),
                        default="auto")
    parser.add_argument("--skip-rows", type=int, default=0,
                        help="Number of header rows in an edge-list file")
    parser.add_argument("--directed", action="store_true",
                        help="Keep edge direction; by default edges are treated as undirected")
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--walk-length", type=int, default=10)
    parser.add_argument("--walks-per-node", type=int, default=10)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--negative", type=int, default=5)
    parser.add_argument("--walk-seed", type=int, default=0)
    parser.add_argument("--word2vec-seed", type=int, default=1)
    args = parser.parse_args()

    if args.dimensions < 1 or args.walk_length < 1 or args.walks_per_node < 1:
        parser.error("dimensions, walk-length, and walks-per-node must be positive")
    if args.window < 1 or args.epochs < 1 or args.negative < 0:
        parser.error("window/epochs must be positive and negative must be non-negative")
    if args.num_nodes is not None and args.num_nodes < 1:
        parser.error("num-nodes must be positive")

    edges, matrix_num_nodes = read_graph(
        args.edges, args.delimiter, args.skip_rows, args.mat_key
    )
    node_count = args.num_nodes if args.num_nodes is not None else matrix_num_nodes
    edges, num_nodes = map_to_row_indices(edges, args.node_ids, node_count)

    adjacency = csr_matrix(
        (np.ones(edges.shape[0], dtype=np.int32), (edges[:, 0], edges[:, 1])),
        shape=(num_nodes, num_nodes),
    )
    adjacency.data[:] = 1
    if not args.directed:
        adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    indptr, neighbors = adjacency.indptr, adjacency.indices

    class WalkCorpus:
        def __iter__(self):
            rng = random.Random(args.walk_seed)
            nodes = list(range(num_nodes))
            for _ in range(args.walks_per_node):
                rng.shuffle(nodes)
                for start in nodes:
                    walk = [start]
                    while len(walk) < args.walk_length:
                        current = walk[-1]
                        choices = neighbors[indptr[current]:indptr[current + 1]]
                        if choices.size == 0:
                            break
                        walk.append(int(rng.choice(choices)))
                    yield [str(node) for node in walk]

    print(f"Nodes: {num_nodes}")
    print(f"Edges in adjacency: {adjacency.nnz}")
    print(f"Generating {args.walks_per_node} walks per node (streamed)...")
    model = Word2Vec(
        sentences=WalkCorpus(),
        vector_size=args.dimensions,
        window=args.window,
        min_count=0,
        sg=1,
        hs=1,
        negative=args.negative,
        workers=1,
        seed=args.word2vec_seed,
        epochs=args.epochs,
    )

    if len(model.wv) != num_nodes:
        raise RuntimeError(f"Expected {num_nodes} node embeddings, got {len(model.wv)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save_word2vec_format(str(args.output), binary=False)
    print(f"Saved {len(model.wv)} embeddings of dimension {model.wv.vector_size} to {args.output}")


if __name__ == "__main__":
    main()
