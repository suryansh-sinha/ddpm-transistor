import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.transforms.functional import gaussian_blur

# Fixed blur parameters
BLUR_KERNEL = 5
BLUR_SIGMA = 2.0

# Must match the README_NOISE_STEP in evaluate_ssim.py
README_NOISE_STEP = 75


def compute_image_scores(anomaly_map):
    """Computes 3 scalar scores from a 2D anomaly map."""
    scores = {}
    scores["mean"] = anomaly_map.mean().item()
    scores["raw_max"] = anomaly_map.max().item()

    map_4d = anomaly_map.unsqueeze(0).unsqueeze(0)
    blurred = gaussian_blur(
        map_4d, kernel_size=[BLUR_KERNEL, BLUR_KERNEL], sigma=[BLUR_SIGMA, BLUR_SIGMA]
    ).squeeze()
    k = max(1, int(blurred.numel() * 0.01)) # numel: returns total number of elements in blurred
    topk_vals, _ = torch.topk(blurred.flatten(), k)
    scores["smoothed_top_1"] = topk_vals.mean().item()

    return scores


def main():
    eval_dir = Path("./eval_outputs")
    if not eval_dir.exists():
        print("eval_outputs not found. Run evaluate_ssim.py first.")
        return

    t_dirs = sorted(
        [d for d in eval_dir.iterdir() if d.is_dir() and d.name.startswith("t_")]
    )
    all_results = []

    print("Starting scoring loop...")
    for t_dir in t_dirs:
        t_val = int(t_dir.name.split("_")[1])

        with open(t_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        image_scores = {r: [] for r in ["mean", "raw_max", "smoothed_top_1"]}
        labels = []
        defect_types_list = []
        all_anomaly_pixels = []
        all_mask_pixels = []

        for item in metadata:
            ssim_map = torch.load(item["map_path"], map_location="cpu")
            mask = torch.load(item["mask_path"], map_location="cpu").squeeze(0)
            mask = (mask > 0.5).float()

            anomaly_map = 1.0 - ssim_map

            # Catches metadata/map misalignment instantly
            expected_mean = 1.0 - item["ssim_score"]
            assert abs(anomaly_map.mean().item() - expected_mean) < 1e-4, (
                f"metadata/map alignment failed for idx {item['dataset_idx']}"
            )

            labels.append(item["label"])
            defect_types_list.append(item["defect_type"])

            scores = compute_image_scores(anomaly_map)
            for r, val in scores.items():
                image_scores[r].append(val)

            all_anomaly_pixels.append(anomaly_map.flatten())
            all_mask_pixels.append(mask.flatten())

        # Image AUROC
        for r in image_scores.keys():
            auroc = roc_auc_score(labels, image_scores[r])
            all_results.append(
               {"t": t_val, "level": "image", "reduction": r, "defect": "all", "auroc": auroc}
            )

        # Per-Defect-Type Image AUROC (Good vs. Specific Defect)
        unique_defects = sorted(list(set([d for d in defect_types_list if d != "good"])))
        for defect in unique_defects:
            idxs = [i for i, d in enumerate(defect_types_list) if d == defect or d == "good"]
            y_true = [labels[i] for i in idxs]
            if len(set(y_true)) < 2:
                continue
            for r in image_scores.keys():
                y_score = [image_scores[r][i] for i in idxs]
                auroc = roc_auc_score(y_true, y_score)
                all_results.append(
                    {"t": t_val, "level": "image", "reduction": r, "defect": defect, "auroc": auroc}
                )

        # Pixel AUROC (Includes good images to penalize false alarms)
        flat_anom = torch.cat(all_anomaly_pixels).numpy()
        flat_mask = torch.cat(all_mask_pixels).numpy()
        pixel_auroc = roc_auc_score(flat_mask, flat_anom)
        all_results.append(
            {"t": t_val, "level": "pixel", "reduction": "map", "auroc": pixel_auroc}
        )

    df = pd.DataFrame(all_results)
    csv_path = eval_dir / "summary_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    img_df_all = df[(df["level"] == "image") & (df["defect"] == "all")]
    best_row = img_df_all.loc[img_df_all["auroc"].idxmax()]
    best_t = int(best_row["t"])

    # This is the best score found by sweeping t and reductions on the test set.
    # In a strict production setting, these hyperparameters would be tuned on a validation set.
    print("\n" + "=" * 60)
    print("BEST IMAGE AUROC (All defects, Best-of-grid on test set)")
    print(f"   Noise Step (t): {best_t}")
    print(f"   Reduction:      {best_row['reduction']}")
    print(f"   AUROC:          {best_row['auroc']:.4f}")
    print("=" * 60 + "\n")

    # Plot: Image AUROC vs t (Overall)
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in ["mean", "raw_max", "smoothed_top_1"]:
        subset = img_df_all[img_df_all["reduction"] == r].sort_values("t")
        ax.plot(subset["t"], subset["auroc"], marker="o", label=r)

    ax.set_xlabel("Noise Step (t)")
    ax.set_ylabel("Image AUROC")
    ax.set_title("Image-Level AUROC vs Noise Step (All Defects)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(eval_dir / "image_auroc_vs_t.png", dpi=150)
    print("Saved image_auroc_vs_t.png")

    # Plot: Qualitative Samples
    # Explicitly use README_NOISE_STEP (200) for the qualitative
    # figure instead of the best-scoring t. This is because I specifically saved
    # the denoised reconstructions for t=75 in evaluate_ssim.py, to show exactly what
    # the model "fixed" (Original -> Denoised -> Anomaly Map).
    print(
        f"\nGenerating qualitative samples using t={README_NOISE_STEP} to include denoised reconstructions..."
    )

    qual_t_dir = eval_dir / f"t_{README_NOISE_STEP}"
    with open(qual_t_dir / "metadata.json", "r") as f:
        qual_meta = json.load(f)

    # Select qualitative samples: first image of each defect + 2 good
    target_defects = ["bent_lead", "cut_lead", "damaged_case", "misplaced"]
    qual_samples = []
    found_defects = {d: False for d in target_defects}
    good_count = 0
    for item in qual_meta:
        dtype = item["defect_type"]
        if dtype != "good" and not found_defects[dtype]:
            qual_samples.append(item)
            found_defects[dtype] = True
        elif dtype == "good" and good_count < 2:
            qual_samples.append(item)
            good_count += 1
        if len(qual_samples) == 6:
            break

    meta_lookup = {item["dataset_idx"]: item for item in qual_meta}

    # 5 columns: Original, Denoised, GT Mask, Raw Anomaly, Smoothed Anomaly
    fig, axes = plt.subplots(len(qual_samples), 5, figsize=(20, 4 * len(qual_samples)))
    if len(qual_samples) == 1:
        axes = axes.reshape(1, -1)

    for i, sample in enumerate(qual_samples):
        item = meta_lookup[sample["dataset_idx"]]

        # Original Image
        img_raw = decode_image(item["path"]).float() / 255.0
        img_orig = v2.Resize((256, 256))(img_raw).permute(1, 2, 0).numpy()

        # Denoised Image (Saved at t=75 in [-1, 1] range, convert to [0, 1])
        denoised = torch.load(item["denoised_path"], map_location="cpu")
        denoised_np = (denoised + 1.0) / 2.0
        denoised_np = denoised_np.permute(1, 2, 0).numpy()

        # Ground Truth Mask
        mask = torch.load(item["mask_path"], map_location="cpu").squeeze(0).numpy()

        # Anomaly Maps
        ssim_map = torch.load(item["map_path"], map_location="cpu")
        anomaly_map = 1.0 - ssim_map
        map_4d = anomaly_map.unsqueeze(0).unsqueeze(0)
        blurred = (
            gaussian_blur(
                map_4d,
                kernel_size=[BLUR_KERNEL, BLUR_KERNEL],
                sigma=[BLUR_SIGMA, BLUR_SIGMA],
            )
            .squeeze()
            .numpy()
        )

        axes[i, 0].imshow(img_orig)
        axes[i, 0].set_title(f"Original ({item['defect_type']})")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(denoised_np)
        axes[i, 1].set_title(f"Denoised (t={README_NOISE_STEP})")
        axes[i, 1].axis("off")
        axes[i, 2].imshow(mask, cmap="gray")
        axes[i, 2].set_title("GT Mask")
        axes[i, 2].axis("off")
        axes[i, 3].imshow(anomaly_map.numpy(), cmap="jet")
        axes[i, 3].set_title("Raw Anomaly")
        axes[i, 3].axis("off")
        axes[i, 4].imshow(blurred, cmap="jet")
        axes[i, 4].set_title("Smoothed Anomaly")
        axes[i, 4].axis("off")

    plt.tight_layout()
    plt.savefig(eval_dir / "qualitative_samples.png", dpi=150)
    print("Saved qualitative_samples.png\nScoring complete!")


if __name__ == "__main__":
    main()
