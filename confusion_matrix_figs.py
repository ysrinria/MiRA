import numpy as np
import matplotlib.pyplot as plt

# classes = ["Disgust", "Sad", "Neutral", "Surprise", "Angry", "Happy", "Fear"] # DFEW

# classes = ["Contempt", "Anxiety", "Neutral", "Sadness", "Anger", "Disgust", "Fear", "Surprise", "Happiness", "Helplessness", "Disappointment"] # MAFW 

classes = ["Disgust", "Sad", "Neutral", "Surprise", "Angry", "Happy", "Fear"] # FERV39k


cm_text = """
105 29 112 25 125 65 6
30 826 257 24 154 77 25
26 215 1284 49 191 182 11
10 70 164 173 114 79 28
26 207 341 55 785 56 17
18 55 248 27 56 1064 5
7 139 105 30 72 29 49
"""

cm = np.array([
    list(map(int, line.split()))
    for line in cm_text.strip().splitlines()
])

cm_norm = cm / cm.sum(axis=1, keepdims=True)
recalls = np.diag(cm) / cm.sum(axis=1)
uar = recalls.mean()
war = np.trace(cm) / cm.sum()

fig, ax = plt.subplots(figsize=(12, 11))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Row-normalized (%)", fontsize=18)
cbar.ax.tick_params(labelsize=14)

ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=16)
ax.set_yticklabels(classes, fontsize=16)

ax.set_xlabel("Predicted Label", fontsize=20, fontweight="bold")
ax.set_ylabel("Ground Truth Label", fontsize=20, fontweight="bold")
ax.set_title("MAFW Confusion Matrix", fontsize=26, fontweight="bold")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax.text(
            j, i,
            f"{cm_norm[i,j]*100:.1f}%\n({cm[i,j]})",
            ha="center", va="center",
            fontsize=11,
            color=color,
            fontweight="bold"
        )

recall_line = "   |   ".join(
    [f"{cls}: {r*100:.2f}%" for cls, r in zip(classes, recalls)]
)

summary_line = f"UAR: {uar*100:.2f}%     |     WAR: {war*100:.2f}%"

fig.text(0.5, 0.105, "Per-class Recall (Row-wise)", ha="center",
         fontsize=18, fontweight="bold")
fig.text(0.5, 0.07, recall_line, ha="center",
         fontsize=15, fontweight="bold")
fig.text(0.5, 0.03, summary_line, ha="center",
         fontsize=20, fontweight="bold")

plt.tight_layout(rect=[0, 0.16, 1, 1])
plt.savefig("FERV39k_LARGEflash_confusion_matrix.png", dpi=300, bbox_inches="tight")
# plt.savefig("confusion_matrix.pdf", bbox_inches="tight")

print(f"Saved confusion_matrix.png / confusion_matrix.pdf")
print(f"UAR: {uar*100:.2f}%")
print(f"WAR: {war*100:.2f}%")