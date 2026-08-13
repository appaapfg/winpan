from ultralytics import YOLO

from pathlib import Path
import json
import numpy as np

"""Validates Ultralytics YOLO models and saves aggregated metrics.

This script loads a list of YOLO models, validates them on a specified dataset,
and saves the results in CSV, JSON, and text formats. It computes aggregated
'all' metrics for overall performance across all classes.

Args:
    None (script-level constants are used for models, paths, and data).

Returns:
    None. Outputs are saved to files in the specified project directory.

Raises:
    Various exceptions from Ultralytics validation or file I/O operations.

Examples:
    Run the script directly: python validate_ultralytics_models.py
"""

# List of model filenames to validate (relative to PREFIX).
MODELS = [
    "yoloe-11s-seg.pt",
    "yoloe-11m-seg.pt",
    "yoloe-11l-seg.pt",
    "yolo11l_panopticon_a0.pt",
    "yolo11_pan_detector_b0.pt",
    "yolo11_pan_detector_b2_1_ft.pt",
    "yolo11_pan_detector_b1.pt",
    "yolo11_pan_detector_b2.pt",
    "yolo11_pan_detector_b3.pt",
    "yolo11_pan_detector_b3_1_large.pt",
    "yolo11_pan_detector_b3_1_medium.pt",
    "yolo11_pan_detector_b3_1_small.pt",
]
PREFIX = "/home/pan/....../"
DATA = "/home/pan/Desktop/yolo_ds_B3/data.yaml"
for MODEL_NAME in MODELS:
    # Process each model in the list for validation.
    if MODEL_NAME.startswith("yoloe"):
        model = YOLO(MODEL_NAME.replace('-seg.pt', '.yaml')).load(PREFIX + MODEL_NAME)
    else:
        model = YOLO(PREFIX + MODEL_NAME)

    # Validate the model on the test split with specified parameters.
    metrics = model.val(
        data=DATA,
        device="cuda",
        batch=1,
        workers=2,
        # Parameters for testing
        imgsz=1280,
        split="test",
        # Parameters for saving
        project=PREFIX + "validation_runs/",
        name=MODEL_NAME.replace('.pt', ''),
        plots=True,
        visualize=True,
        save_json=True
    )

    # Save metrics and summaries to the run folder.
    # Determine the save directory from the metrics object; fallback if not available.
    save_dir = Path(metrics.save_dir) if hasattr(metrics, "save_dir") else None
    if save_dir is None:
        # Fallback: construct path from project and name parameters.
        save_dir = Path("./documents/validation_runs") / (PREFIX + "valid_runs" + MODEL_NAME.replace('.pt', ''))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare an aggregated 'all' entry for CSV + JSON outputs (overall metrics across classes).
    decimals = 5  # Number of decimal places for rounding metric values.
    # Compute number of unique images containing targets.
    try:
        img_ids = np.asarray(metrics.stats.get("target_img", []))
        images = int(len(np.unique(img_ids))) if img_ids.size else 0
    except Exception:
        images = 0
    # Compute total number of target instances across all classes.
    try:
        instances = int(np.sum(metrics.nt_per_class))
    except Exception:
        try:
            instances = int(len(np.asarray(metrics.stats.get("target_cls", []))))
        except Exception:
            instances = 0

    # Extract box-level metrics (precision, recall, mAP) when available.
    try:
        mp = float(metrics.box.mp)
        mr = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
    except Exception:
        try:
            mean_results = list(metrics.mean_results())
            mp, mr, map50, map50_95 = [float(x) for x in mean_results[:4]]
        except Exception:
            mp = mr = map50 = map50_95 = 0.0

    box_f1 = round((2 * mp * mr / (mp + mr)) if (mp + mr) else 0.0, decimals)

    all_entry = {
        "Class": "all",
        "Images": images,
        "Instances": instances,
        "Box-P": round(mp, decimals),
        "Box-R": round(mr, decimals),
        "Box-F1": box_f1,
        "mAP50": round(map50, decimals),
        "mAP50-95": round(map50_95, decimals),
    }
    try:
        if hasattr(metrics, "seg"):
            mp_m = float(metrics.seg.mp)
            mr_m = float(metrics.seg.mr)
            map50_m = float(metrics.seg.map50)
            map50_95_m = float(metrics.seg.map)
            mask_f1 = round((2 * mp_m * mr_m / (mp_m + mr_m)) if (mp_m + mr_m) else 0.0, decimals)
            all_entry.update(
                {
                    "Mask-P": round(mp_m, decimals),
                    "Mask-R": round(mr_m, decimals),
                    "Mask-F1": mask_f1,
                    "mAP50(M)": round(map50_m, decimals),
                    "mAP50-95(M)": round(map50_95_m, decimals),
                }
            )
    except Exception:
        pass

    # Write CSV using polars DataFrame (metrics.to_df()) if available, else write the CSV string
    df = metrics.to_df()
    csv_path = save_dir / "metrics.csv"
    if hasattr(df, "write_csv"):
        # Polars DataFrame: we append the 'all' row by converting to pandas and inserting, or directly to CSV
        try:
            import polars as pl

            # Create a single-row DataFrame with the same columns as `df`
            # Convert df to pandas if easier to append; polars has vstack but need to create the correct dtype
            try:
                # Polars: create a DataFrame from a dictionary using the same column order
                cols = df.columns
                # ensure all keys exist in all_entry (set None for missing)
                row = {c: all_entry.get(c, None) for c in cols}
                all_df = pl.DataFrame([row])
                df = df.vstack(all_df)
            except Exception:
                # fallback: just write existing df, then append CSV row text at the end
                df.write_csv(csv_path)
                with open(csv_path, "a") as f:
                    f.write("\n")
                    f.write(",".join([str(all_entry.get(c, "")) for c in df.columns]))
            else:
                df.write_csv(csv_path)
        except Exception:
            # fallback if polars isn't available: write df and append a raw CSV line
            df.write_csv(csv_path)
            with open(csv_path, "a") as f:
                f.write("\n")
                f.write(",".join([str(all_entry.get(c, "")) for c in df.columns]))
    else:
        csv_str = metrics.to_csv()
        # append all row to the csv string
        try:
            import csv as _csv
            from io import StringIO

            sio = StringIO(csv_str)
            reader = list(_csv.reader(sio))
            headers = reader[0]
            all_row = [str(all_entry.get(h, "")) for h in headers]
            csv_str = csv_str + "\n" + ",".join(all_row)
        except Exception:
            # just write csv_str
            pass
        csv_path.write_text(csv_str)
    print(f"Saved metrics CSV to {csv_path}")

    # Save JSON representation using metrics.to_json()
    json_str = metrics.to_json()
    # Try to include an aggregated 'all' metrics row in JSON
    # If the JSON is a list of per-class dictionaries, we'll insert 'all' as the first row
    try:
        entries = json.loads(json_str)
        # Only proceed if it's a list of dicts
        if isinstance(entries, list) and entries and isinstance(entries[0], dict):
            # compute aggregated metrics (robust to segment/detection)
            decimals = 5
            # Images (unique images containing targets)
            images = 0
            instances = 0
            try:
                # metrics.stats['target_img'] may be a numpy array or list
                img_ids = np.asarray(metrics.stats["target_img"])
                images = int(len(np.unique(img_ids)))
            except Exception:
                try:
                    # fallback: count number of images in nt_per_image non-zero entries
                    images = int(np.count_nonzero(metrics.nt_per_image))
                except Exception:
                    images = 0
            # Instances: use number of target instances
            try:
                instances = int(np.sum(metrics.nt_per_class))
            except Exception:
                try:
                    instances = int(len(np.asarray(metrics.stats["target_cls"])))
                except Exception:
                    instances = 0

            # Box metrics (when available)
            try:
                mp = float(metrics.box.mp)
                mr = float(metrics.box.mr)
                map50 = float(metrics.box.map50)
                map50_95 = float(metrics.box.map)
            except Exception:
                # fallback to mean_results (first 4 elements are box results)
                try:
                    mean_results = list(metrics.mean_results())
                    mp, mr, map50, map50_95 = [float(x) for x in mean_results[:4]]
                except Exception:
                    mp = mr = map50 = map50_95 = 0.0

            box_f1 = round((2 * mp * mr / (mp + mr)) if (mp + mr) else 0.0, decimals)

            all_entry = {
                "Class": "all",
                "Images": images,
                "Instances": instances,
                "Box-P": round(mp, decimals),
                "Box-R": round(mr, decimals),
                "Box-F1": box_f1,
                "mAP50": round(map50, decimals),
                "mAP50-95": round(map50_95, decimals),
            }

            # If segmentation metrics present, append mask metrics as well
            try:
                if hasattr(metrics, "seg"):
                    mp_m = float(metrics.seg.mp)
                    mr_m = float(metrics.seg.mr)
                    map50_m = float(metrics.seg.map50)
                    map50_95_m = float(metrics.seg.map)
                    mask_f1 = round((2 * mp_m * mr_m / (mp_m + mr_m)) if (mp_m + mr_m) else 0.0, decimals)
                    all_entry.update(
                        {
                            "Mask-P": round(mp_m, decimals),
                            "Mask-R": round(mr_m, decimals),
                            "Mask-F1": mask_f1,
                            "mAP50(M)": round(map50_m, decimals),
                            "mAP50-95(M)": round(map50_95_m, decimals),
                        }
                    )
            except Exception:
                pass

            # If there's already an 'all' entry, replace it; otherwise insert at the top
            idx_all = next((i for i, e in enumerate(entries) if isinstance(e, dict) and e.get("Class") == "all"), None)
            if idx_all is not None:
                entries[idx_all] = all_entry
            else:
                entries.insert(0, all_entry)
            json_str = json.dumps(entries, indent=2)
    except Exception:
        # If something goes wrong parsing the JSON, fall back to the original JSON
        pass
    metrics_json_path = save_dir / f"{MODEL_NAME.replace('.pt','')}_metrics.json"
    metrics_json_path.write_text(json_str)
    print(f"Saved metrics JSON to {metrics_json_path}")

    # Save plain text summary too
    summary = metrics.summary()
    (save_dir / "metrics_summary.txt").write_text(str(summary))
    print(f"Saved metrics summary to {save_dir / 'metrics_summary.txt'}")