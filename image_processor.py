"""
image_processor.py - Parallel image processing pipeline (OS course project).

Processes all images under images/<subdir>/ dynamically:
    load -> grayscale -> blur -> edge detection (on blurred) -> save

Concurrency is managed manually with multiprocessing.Process (NOT Pool).
Synchronization uses Queue (task/result passing), Lock (shared logging),
and Semaphore (limit concurrent disk I/O).

Usage:
    python image_processor.py                  # parallel, auto worker count
    python image_processor.py --workers 4      # parallel with 4 workers
    python image_processor.py --sequential     # single-process baseline
    python image_processor.py --benchmark      # run timing study + PDF report
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from multiprocessing import Lock, Process, Queue, Semaphore, cpu_count

from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from image_filters import (
    apply_blur,
    compute_stats,
    detect_edges,
    load_image,
    save_image,
    to_grayscale,
)

# Light blur keeps structural edges detectable (radius 2 over-smooths FIND_EDGES).
BLUR_RADIUS = 1
MAX_CONCURRENT_IO = 2
SENTINEL = None


def discover_batches(images_root="images"):
    """Return sorted subdirectory names under images_root that contain PNG files."""
    if not os.path.isdir(images_root):
        return []
    batches = []
    for name in sorted(os.listdir(images_root)):
        path = os.path.join(images_root, name)
        if not os.path.isdir(path):
            continue
        has_png = any(f.lower().endswith(".png") for f in os.listdir(path))
        if has_png:
            batches.append(name)
    return batches


def collect_image_tasks(images_root="images", output_root="output"):
    """Return (input_path, output_path) pairs for every PNG in each batch subdir."""
    tasks = []
    for batch in discover_batches(images_root):
        input_dir = os.path.join(images_root, batch)
        for name in sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".png")):
            src = os.path.join(input_dir, name)
            dst = os.path.join(output_root, batch, name)
            tasks.append((src, dst))
    return tasks


def process_image_pipeline(input_path, output_path, blur_radius=BLUR_RADIUS):
    """
    Run the full filter pipeline on one image.

    Returns a dict with the source path and computed statistics.
    """
    img = load_image(input_path)
    gray = to_grayscale(img)
    blurred = apply_blur(gray, radius=blur_radius)
    edges = detect_edges(blurred)  # edges from blurred grayscale, not original
    save_image(edges, output_path)
    stats = compute_stats(img)
    return {
        "input_path": input_path,
        "output_path": output_path,
        "filename": os.path.basename(input_path),
        "batch": os.path.basename(os.path.dirname(input_path)),
        **stats,
    }


def worker(task_queue, result_queue, print_lock, io_semaphore):
    """
    Consumer process: pull tasks until sentinel, process, push results.

    io_semaphore caps how many workers hit disk at once.
    print_lock keeps console lines from interleaving.
    """
    while True:
        task = task_queue.get()
        if task is SENTINEL:
            break

        input_path, output_path = task
        with io_semaphore:
            result = process_image_pipeline(input_path, output_path)

        result_queue.put(result)
        with print_lock:
            print(f"  [worker {os.getpid()}] {result['batch']}/{result['filename']}")


def run_sequential(tasks):
    """Process every image in the main process (baseline for speedup)."""
    results = []
    start = time.perf_counter()
    for input_path, output_path in tasks:
        result = process_image_pipeline(input_path, output_path)
        results.append(result)
        print(f"  [sequential] {result['batch']}/{result['filename']}")
    elapsed = time.perf_counter() - start
    return elapsed, results


def run_parallel(tasks, num_workers):
    """
    Distribute tasks across manually spawned worker processes.

    Each worker reads from a shared task Queue and writes results to a
    result Queue. Sentinels (None) tell workers to exit cleanly.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    task_queue = Queue()
    result_queue = Queue()
    print_lock = Lock()
    io_semaphore = Semaphore(MAX_CONCURRENT_IO)

    for task in tasks:
        task_queue.put(task)
    for _ in range(num_workers):
        task_queue.put(SENTINEL)

    workers = [
        Process(
            target=worker,
            args=(task_queue, result_queue, print_lock, io_semaphore),
        )
        for _ in range(num_workers)
    ]

    start = time.perf_counter()
    for proc in workers:
        proc.start()

    results = []
    finished = 0
    while finished < len(tasks):
        results.append(result_queue.get())
        finished += 1

    for proc in workers:
        proc.join()

    elapsed = time.perf_counter() - start
    return elapsed, results


def write_stats_csv(results, path="output/image_stats.csv"):
    """Persist per-image statistics collected during processing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "batch",
        "filename",
        "input_path",
        "output_path",
        "mean_brightness",
        "std_brightness",
        "edge_density",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(results, key=lambda r: (r["batch"], r["filename"])):
            writer.writerow({k: row[k] for k in fieldnames})
    return path


def run_benchmark(tasks, worker_counts=None):
    """
    Time sequential and parallel runs; return rows for the analysis report.

    Parallel runs use a temporary output directory so benchmarks do not
    overwrite each other's files.
    """
    if worker_counts is None:
        cores = cpu_count() or 1
        worker_counts = sorted({1, 2, 4, cores})

    rows = []

    print("\n=== Benchmark: sequential baseline ===")
    seq_elapsed, _ = run_sequential(tasks)
    rows.append(
        {
            "mode": "sequential",
            "workers": 1,
            "elapsed_sec": round(seq_elapsed, 4),
            "speedup": 1.0,
            "efficiency_pct": 100.0,
        }
    )
    print(f"  Time: {seq_elapsed:.3f}s\n")

    for n in worker_counts:
        if n == 1:
            continue
        print(f"=== Benchmark: {n} workers ===")
        bench_tasks = [
            (src, dst.replace("output", "output_benchmark"))
            for src, dst in tasks
        ]
        elapsed, _ = run_parallel(bench_tasks, n)
        speedup = seq_elapsed / elapsed if elapsed > 0 else 0.0
        efficiency = (speedup / n) * 100.0
        rows.append(
            {
                "mode": "parallel",
                "workers": n,
                "elapsed_sec": round(elapsed, 4),
                "speedup": round(speedup, 3),
                "efficiency_pct": round(efficiency, 1),
            }
        )
        print(f"  Time: {elapsed:.3f}s  speedup: {speedup:.2f}x  efficiency: {efficiency:.1f}%\n")

    return rows, seq_elapsed


def _fmt_edge_pct(edge_density):
    """Format edge density fraction as a percentage string (e.g. 0.0463 -> '4.6%')."""
    return f"{edge_density * 100:.1f}%"


def write_analysis_summary(results, path="reports/analysis_summary.txt"):
    """
    Write analysis_summary.txt in the project-required tabular layout:

      BATCHES PROCESSED
      per-batch tables (Filename / Brightness / Contrast / Edge Density)
      BATCH AVERAGE row per batch
      SUMMARY ending with 'Processing complete.'
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    batches = {}
    for row in results:
        batches.setdefault(row["batch"], []).append(row)

    batch_names = sorted(batches.keys())
    total_images = len(results)

    # Column layout matching the required Output Example
    col_fmt = "{:<16} {:>12} {:>12} {:>14}"
    header = col_fmt.format("Filename", "Brightness", "Contrast", "Edge Density")
    divider = "-" * len(header)

    lines = [
        "IMAGE ANALYSIS SUMMARY",
        "=" * len(header),
        "",
        "BATCHES PROCESSED",
        "-" * 17,
    ]
    for idx, batch in enumerate(batch_names, start=1):
        lines.append(f"  {idx}. {batch}")
    lines.append(f"Total images: {total_images}")
    lines.append("")

    batch_avgs = {}
    for batch in batch_names:
        items = sorted(batches[batch], key=lambda r: r["filename"])
        count = len(items)
        avg_mean = sum(r["mean_brightness"] for r in items) / count
        avg_std = sum(r["std_brightness"] for r in items) / count
        avg_edge = sum(r["edge_density"] for r in items) / count
        batch_avgs[batch] = {
            "mean_brightness": avg_mean,
            "std_brightness": avg_std,
            "edge_density": avg_edge,
        }

        lines.append(f"BATCH: {batch}")
        lines.append(header)
        lines.append(divider)
        for row in items:
            lines.append(
                col_fmt.format(
                    row["filename"],
                    f"{row['mean_brightness']:.2f}",
                    f"{row['std_brightness']:.2f}",
                    _fmt_edge_pct(row["edge_density"]),
                )
            )
        lines.append(divider)
        lines.append(
            col_fmt.format(
                "BATCH AVERAGE",
                f"{avg_mean:.2f}",
                f"{avg_std:.2f}",
                _fmt_edge_pct(avg_edge),
            )
        )
        lines.append("")

    brightest = max(batch_avgs, key=lambda b: batch_avgs[b]["mean_brightness"])
    darkest = min(batch_avgs, key=lambda b: batch_avgs[b]["mean_brightness"])
    highest_edge = max(batch_avgs, key=lambda b: batch_avgs[b]["edge_density"])

    lines.extend(
        [
            "SUMMARY",
            "-" * 7,
            f"Brightest batch: {brightest}",
            f"Darkest batch: {darkest}",
            f"Highest edge density: {highest_edge}",
            "",
            "Processing complete.",
            "",
        ]
    )

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def create_comparison_image(
    batch_name,
    images_root="images",
    output_root="output",
    save_path="reports/comparison_sample.png",
    n_cols=3,
):
    """Build a before/after grid (original vs edge-detected) for embedding in the PDF."""
    input_dir = os.path.join(images_root, batch_name)
    edge_dir = os.path.join(output_root, batch_name)
    if not os.path.isdir(input_dir) or not os.path.isdir(edge_dir):
        return None

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".png"))[:n_cols]
    if not files:
        return None

    sample = Image.open(os.path.join(input_dir, files[0]))
    width, height = sample.size
    pad, label_h = 8, 22
    cols = len(files)
    grid_w = pad + cols * (width + pad)
    grid_h = pad + 2 * (label_h + height + pad)
    grid = Image.new("RGB", (grid_w, grid_h), color=(40, 40, 40))
    draw = ImageDraw.Draw(grid)

    for row, label in enumerate(["Original", "Edge Detection (from blurred)"]):
        y_label = pad + row * (label_h + height + pad)
        y_img = y_label + label_h
        draw.text((pad, y_label), f"-- {label} ({batch_name}) --", fill=(200, 200, 200))
        for col, filename in enumerate(files):
            x = pad + col * (width + pad)
            if row == 0:
                tile = Image.open(os.path.join(input_dir, filename)).convert("RGB")
            else:
                edge_path = os.path.join(edge_dir, filename)
                tile = (
                    Image.open(edge_path).convert("RGB")
                    if os.path.exists(edge_path)
                    else Image.new("RGB", (width, height), color=(80, 40, 40))
                )
            grid.paste(tile, (x, y_img))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    grid.save(save_path)
    return save_path


def _pdf_table(data, col_widths=None):
    table = Table(data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F2F2")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def write_analysis_report(
    benchmark_rows,
    seq_time,
    num_images,
    batch_names,
    comparison_image=None,
    path="ANALYSIS_REPORT.pdf",
):
    """Generate the written analysis required by the OS project as a PDF."""
    cores = cpu_count() or 1
    num_workers = cores
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], spaceAfter=8, spaceBefore=14)
    body = styles["BodyText"]

    doc = SimpleDocTemplate(path, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []

    story.append(Paragraph("Operating Systems Project — Performance Analysis Report", styles["Title"]))
    story.append(
        Paragraph(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"])
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("1. System Configuration", heading))
    story.append(
        _pdf_table(
            [
                ["Parameter", "Value"],
                ["OS", f"{platform.system()} {platform.release()}"],
                ["Python", platform.python_version()],
                ["CPU cores (logical)", str(cores)],
                ["Worker processes used", str(num_workers)],
                ["Images processed", str(num_images)],
                ["Batches discovered", ", ".join(batch_names) if batch_names else "(none)"],
                [
                    "Pipeline",
                    f"load -> grayscale -> blur (r={BLUR_RADIUS}) -> "
                    "edge detection on blurred -> save",
                ],
            ],
            col_widths=[2.2 * inch, 4.3 * inch],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("2. Design Justifications", heading))

    story.append(Paragraph("<b>Why multiprocessing instead of threading?</b>", body))
    story.append(
        Paragraph(
            "Image filtering in Pillow is CPU-bound Python work. The Global Interpreter Lock "
            "(GIL) prevents multiple threads from executing Python bytecode in parallel, so "
            "threads would not speed up the blur and edge-detection steps. Separate processes "
            "each have their own interpreter and can run on different CPU cores at the same time.",
            body,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph("<b>Why this worker count?</b>", body))
    story.append(
        Paragraph(
            f"The default configuration uses {num_workers} worker processes, matching the "
            f"{cores} logical CPU core(s) reported by the system. This aims to keep each core "
            "busy without spawning so many processes that scheduling and IPC overhead dominate. "
            "The --workers flag allows tuning when tasks are heavier or I/O-bound.",
            body,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph("<b>Why Queue, Lock, and Semaphore?</b>", body))
    story.append(
        Paragraph(
            f"<b>Queue</b> — safely distributes (input, output) task pairs from the main "
            "process to workers and collects result dictionaries without manual shared-memory "
            "management. Sentinel values (None) signal workers to exit cleanly.<br/>"
            "<b>Lock</b> — serializes console logging so lines from parallel workers do not "
            "interleave and become unreadable.<br/>"
            f"<b>Semaphore({MAX_CONCURRENT_IO})</b> — limits how many workers perform disk "
            "reads/writes at once, reducing I/O contention when many processes share one disk.",
            body,
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("3. Concurrency Primitives", heading))
    story.append(
        _pdf_table(
            [
                ["Primitive", "Role"],
                ["Process", "Workers run the CPU-bound filter pipeline in parallel"],
                ["Queue", "Task distribution and result collection between processes"],
                ["Lock", "Protects shared console output"],
                [f"Semaphore({MAX_CONCURRENT_IO})", "Caps concurrent disk I/O operations"],
            ],
            col_widths=[1.6 * inch, 4.9 * inch],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    if comparison_image and os.path.exists(comparison_image):
        story.append(Paragraph("4. Visual Comparison (Original vs Edge-Detected)", heading))
        story.append(
            Paragraph(
                "Sample images from the first discovered batch showing the original input "
                "alongside the processed edge-detection output.",
                body,
            )
        )
        img = RLImage(comparison_image)
        max_w = 6.5 * inch
        scale = min(1.0, max_w / img.drawWidth)
        img.drawWidth *= scale
        img.drawHeight *= scale
        story.append(img)
        story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("5. Benchmark Results", heading))
    story.append(
        _pdf_table(
            [["Mode", "Workers", "Time (s)", "Speedup", "Efficiency (%)"]]
            + [
                [
                    row["mode"],
                    row["workers"],
                    row["elapsed_sec"],
                    row["speedup"],
                    row["efficiency_pct"],
                ]
                for row in benchmark_rows
            ]
        )
    )
    ms_per_image = (seq_time / num_images * 1000) if num_images else 0
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        Paragraph(
            f"<b>Sequential baseline:</b> {seq_time:.3f} s for {num_images} images "
            f"({ms_per_image:.1f} ms/image).",
            body,
        )
    )

    best = max(
        (r for r in benchmark_rows if r["mode"] == "parallel"),
        key=lambda r: r["speedup"],
        default=None,
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("6. Performance Analysis", heading))
    if best:
        if best["speedup"] >= 1.0:
            story.append(
                Paragraph(
                    f"Best parallel configuration: {best['workers']} workers "
                    f"({best['speedup']}x speedup, {best['efficiency_pct']}% efficiency).",
                    body,
                )
            )
        else:
            story.append(
                Paragraph(
                    f"Parallel runs were slower than sequential on this machine "
                    f"(best: {best['workers']} workers at {best['speedup']}x of baseline). "
                    "Process spawn and Queue IPC overhead can exceed compute savings for small "
                    "images, especially on Windows where the default start method is spawn.",
                    body,
                )
            )
    else:
        story.append(Paragraph("Only sequential timing was recorded.", body))

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("7. Output Files", heading))
    for line in (
        "Processed edge images: output/<batch>/",
        "Per-image statistics: output/image_stats.csv",
        "Analysis summary: reports/analysis_summary.txt",
        "Raw benchmark JSON: output/benchmark_results.json",
    ):
        # Escape XML special chars for reportlab Paragraph; use separate lines (not <br/>)
        story.append(Paragraph(line.replace("<", "&lt;").replace(">", "&gt;"), body))

    doc.build(story)
    return path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Parallel image processing pipeline")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: number of CPU cores)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run in a single process (no parallelism)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run timing study and write ANALYSIS_REPORT.pdf",
    )
    return parser.parse_args(argv)


def main(argv=None):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args(argv)

    batch_names = discover_batches()
    tasks = collect_image_tasks()
    if not tasks:
        print("No images found under images/<subdir>/")
        sys.exit(1)

    print(f"Found {len(tasks)} images across {', '.join(batch_names)}")

    if args.benchmark:
        rows, seq_time = run_benchmark(tasks)
        os.makedirs("output", exist_ok=True)
        with open("output/benchmark_results.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)

    print("\n=== Production run (writes to output/) ===")
    if args.sequential:
        elapsed, results = run_sequential(tasks)
        mode = "sequential"
    else:
        num_workers = args.workers or cpu_count() or 1
        print(f"Starting {num_workers} worker process(es)...")
        elapsed, results = run_parallel(tasks, num_workers)
        mode = f"parallel ({num_workers} workers)"

    stats_path = write_stats_csv(results)
    summary_path = write_analysis_summary(results)

    if args.benchmark:
        comparison = None
        if batch_names:
            comparison = create_comparison_image(batch_names[0])
        report_path = write_analysis_report(
            rows,
            seq_time,
            len(tasks),
            batch_names,
            comparison_image=comparison,
        )
        print(f"Benchmark data  -> output/benchmark_results.json")
        print(f"Analysis report -> {report_path}")

    print(f"\nDone ({mode}) in {elapsed:.3f}s")
    print(f"  Stats CSV : {stats_path}")
    print(f"  Summary   : {summary_path}")
    print(f"  Processed : {len(results)} images -> output/<batch>/")
    print("\nNext: python visualize.py")


if __name__ == "__main__":
    main()
