# BYOKG-RAG Examples

Example notebooks demonstrating BYOKG-RAG capabilities.

## Prerequisites

- Python 3.10+
- [AWS CLI configured with credentials](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html) — verify with `aws sts get-caller-identity`
- Amazon Bedrock model access enabled for Claude and Cohere Embed

## Notebooks

| Notebook | Description | AWS Services |
|----------|-------------|--------------|
| `byokg_rag_demo_local_graph.ipynb` | Local graph store demo | Bedrock only |
| `byokg_rag_neptune_analytics_demo.ipynb` | Neptune Analytics demo | Neptune Analytics, Bedrock, S3 |
| `byokg_rag_neptune_analytics_demo_cypher.ipynb` | Cypher-based linking | Neptune Analytics, Bedrock, S3 |
| `byokg_rag_neptune_analytics_embeddings.ipynb` | Embedding indexing | Neptune Analytics, Bedrock, S3 |
| `byokg_rag_neptune_db_cluster_demo.ipynb` | Neptune DB cluster demo | Neptune DB, Bedrock, S3 |

## Automated Testing

Run the local graph notebook end-to-end:

```bash
cd tests
pip install nbformat nbclient ipykernel
python run_notebooks.py
```

To test against the current local source (instead of the released package):

```bash
python run_notebooks.py --local
```

The runner executes each cell individually and produces a per-cell status report:

```
Summary: 29 succeeded, 2 skipped, 0 failed
```

Configuration:
- GPU/CUDA cells are automatically skipped
- `--local` flag pre-installs byokg-rag from local source and skips the remote pip install cell
