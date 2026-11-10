import pandas as pd
import matplotlib.pyplot as plt


results = pd.read_csv("results/results.csv")
classes = pd.read_csv("results/per_class_accuracy.csv")

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# 1. Gaussian noise comparison
noise = results[results["condition"] == "Gaussian Noise"]

plt.figure(figsize=(7, 5))

for model in noise["experiment"].unique():
    data = noise[noise["experiment"] == model].sort_values("severity")

    plt.plot(
        data["severity"],
        data["test_accuracy"],
        marker="o",
        linewidth=2,
        label=model
    )

plt.xlabel("Gaussian Noise Standard Deviation")
plt.ylabel("Test Accuracy (%)")
plt.title("Model Accuracy Under Gaussian Noise")
plt.ylim(0, 100)
plt.xticks([0.0, 0.05, 0.10, 0.20])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(
    "results/noise_robustness.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# 2. Gaussian blur
blur = results[results["condition"] == "Gaussian Blur"].sort_values("severity")

plt.figure(figsize=(7, 5))

plt.plot(
    blur["severity"],
    blur["test_accuracy"],
    marker="o",
    linewidth=2
)

plt.xlabel("Gaussian Blur Sigma")
plt.ylabel("Test Accuracy (%)")
plt.title("Baseline CNN Accuracy Under Gaussian Blur")
plt.ylim(0, 100)
plt.xticks([0.5, 1.0, 1.5])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    "results/blur_robustness.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# 3. Balanced vs imbalanced per-class accuracy

plt.figure(figsize=(10, 5))

x = range(len(classes))
width = 0.35

plt.bar(
    [i - width / 2 for i in x],
    classes["balanced_accuracy"],
    width=width,
    label="Balanced Training"
)

plt.bar(
    [i + width / 2 for i in x],
    classes["imbalanced_accuracy"],
    width=width,
    label="Imbalanced Training"
)

plt.xlabel("CIFAR-10 Class")
plt.ylabel("Test Accuracy (%)")
plt.title("Balanced vs Imbalanced Per-Class Accuracy")

plt.xticks(
    x,
    classes["class"],
    rotation=30,
    ha="right"
)

plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/balanced_vs_imbalanced.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Saved:")
print("  results/noise_robustness.png")
print("  results/blur_robustness.png")
print("  results/imbalanced_per_class_accuracy.png")