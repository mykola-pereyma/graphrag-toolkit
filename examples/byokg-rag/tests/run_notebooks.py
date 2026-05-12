#!/usr/bin/env python3
"""Execute byokg-rag local notebook cell-by-cell with skip logic and per-cell reporting.

Usage:
    cd examples/byokg-rag/tests && python run_notebooks.py

Requires: nbformat, nbclient
    pip install nbformat nbclient
"""

import argparse
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK = "byokg_rag_demo_local_graph.ipynb"

# Cells to skip: cell_index -> reason
SKIP_CELLS = {}

# Patterns that indicate GPU-dependent cells
GPU_PATTERNS = ["cuda", ".to(device)", "torch.device"]


def should_skip_gpu(source):
    """Check if cell source contains GPU-specific code."""
    lower = source.lower()
    for pattern in GPU_PATTERNS:
        if pattern.lower() in lower:
            return True
    return False


def extract_output(cell):
    """Extract cell output text for reporting."""
    parts = []
    for o in cell.get("outputs", []):
        if o.get("output_type") == "stream":
            parts.append(o.get("text", ""))
        elif o.get("output_type") == "execute_result":
            data = o.get("data", {})
            if "text/plain" in data:
                parts.append(data["text/plain"])
        elif o.get("output_type") == "error":
            parts.append("\n".join(o.get("traceback", [])[-3:]))
    text = "".join(parts).strip()
    lines = text.split("\n")[:20]
    return "\n".join(lines) if lines and lines[0] else "(no output)"


def run_notebook(nb_path, work_dir):
    results = []
    nb_name = os.path.basename(nb_path)
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": work_dir}},
    )
    print(f"\n{'=' * 60}\nNOTEBOOK: {nb_name}\n{'=' * 60}", flush=True)

    with client.setup_kernel():
        for cell_idx, cell in enumerate(nb.cells):
            if cell_idx in SKIP_CELLS:
                reason = SKIP_CELLS[cell_idx]
                print(f"  Cell {cell_idx:3d}: SKIPPED  ({reason})", flush=True)
                results.append(dict(
                    notebook=nb_name, cell_index=cell_idx, cell_type=cell.cell_type,
                    status="SKIPPED", output_summary=f"Skipped: {reason}",
                    exec_time_s=0, error=None, source_preview=cell.source[:150],
                ))
                continue

            if cell.cell_type != "code":
                results.append(dict(
                    notebook=nb_name, cell_index=cell_idx, cell_type=cell.cell_type,
                    status="SUCCESS", output_summary="Markdown cell",
                    exec_time_s=0, error=None, source_preview=cell.source[:150],
                ))
                continue

            if should_skip_gpu(cell.source):
                print(f"  Cell {cell_idx:3d}: SKIPPED  (GPU/CUDA dependency)", flush=True)
                results.append(dict(
                    notebook=nb_name, cell_index=cell_idx, cell_type="code",
                    status="SKIPPED", output_summary="Skipped: GPU/CUDA dependency",
                    exec_time_s=0, error=None, source_preview=cell.source[:150],
                ))
                continue

            start = time.time()
            error_detail = None
            try:
                client.execute_cell(cell, cell_idx)
                status = "SUCCESS"
            except CellExecutionError as e:
                status = "FAILED"
                error_detail = str(e)[-800:]
            except Exception as e:
                status = "FAILED"
                error_detail = f"{type(e).__name__}: {str(e)[:500]}"
            elapsed = round(time.time() - start, 2)

            output_summary = extract_output(cell)
            print(f"  Cell {cell_idx:3d}: {status:<9}({elapsed}s)", flush=True)
            if status == "FAILED":
                print(f"           {(error_detail or 'unknown')[:150]}", flush=True)

            results.append(dict(
                notebook=nb_name, cell_index=cell_idx, cell_type="code",
                status=status, output_summary=output_summary,
                exec_time_s=elapsed, error=error_detail,
                source_preview=cell.source[:150],
            ))

    return results


def write_markdown_report(report, path):
    with open(path, "w") as f:
        success = sum(1 for r in report if r["status"] == "SUCCESS")
        failed = sum(1 for r in report if r["status"] == "FAILED")
        skipped = sum(1 for r in report if r["status"] == "SKIPPED")
        f.write("# Notebook Execution Report\n\n")
        f.write(f"| Metric | Count |\n|--------|-------|\n")
        f.write(f"| Total cells | {len(report)} |\n")
        f.write(f"| SUCCESS | {success} |\n| FAILED | {failed} |\n| SKIPPED | {skipped} |\n\n")

        current_nb = None
        for r in report:
            if r["notebook"] != current_nb:
                current_nb = r["notebook"]
                f.write(f"## {current_nb}\n\n")
                f.write("| Cell | Type | Status | Time | Output Summary |\n")
                f.write("|------|------|--------|------|----------------|\n")
            summary = r["output_summary"].replace("\n", " ")[:100]
            f.write(f"| {r['cell_index']} | {r['cell_type']} | {r['status']} | {r['exec_time_s']}s | {summary} |\n")
            if r["error"]:
                f.write(f"\n**Error (Cell {r['cell_index']}):** `{r['error'][:200]}`\n\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Run byokg-rag local notebook")
    parser.add_argument("--output-dir", default="test-results/")
    parser.add_argument("--local", action="store_true", help="Run with local Python (no Docker)")
    args = parser.parse_args()

    work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    nb_path = os.path.join(work_dir, NOTEBOOK)

    if not os.path.exists(nb_path):
        print(f"ERROR: Notebook not found: {nb_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    results = run_notebook(nb_path, work_dir)

    # Write reports
    json_path = os.path.join(args.output_dir, "execution_report.json")
    md_path = os.path.join(args.output_dir, "execution_report.md")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    write_markdown_report(results, md_path)

    # Print summary
    success = sum(1 for r in results if r["status"] == "SUCCESS")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    skipped = sum(1 for r in results if r["status"] == "SKIPPED")
    print(f"\n{'=' * 60}\nFINAL REPORT\n{'=' * 60}")
    print(f"{'Cell':>5} {'Status':<9} {'First Line'}")
    print(f"{'-'*5} {'-'*9} {'-'*60}")
    for r in results:
        line = r["source_preview"].split("\n")[0][:60]
        print(f"{r['cell_index']:5d} {r['status']:<9} {line}")

    print(f"\nSummary: {success} succeeded, {skipped} skipped, {failed} failed")
    print(f"Reports: {json_path}, {md_path}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
