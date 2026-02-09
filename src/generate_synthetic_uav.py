import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# Parameters
# ---------------------------
np.random.seed(42)
rows, cols = 100, 100   # field size

output_data = "../data/synthetic/uav_synthetic.csv"
output_fig = "../outputs/figures/synthetic_indices.png"

os.makedirs("../data/synthetic", exist_ok=True)
os.makedirs("../outputs/figures", exist_ok=True)

# ---------------------------
# Create spatial grid
# ---------------------------
x = np.arange(cols)
y = np.arange(rows)
xx, yy = np.meshgrid(x, y)

# ---------------------------
# Simulate vegetation patterns
# ---------------------------
ndvi = 0.6 + 0.2*np.sin(xx/10) * np.cos(yy/15)
ndvi += np.random.normal(0, 0.02, (rows, cols))
ndvi = np.clip(ndvi, 0.2, 0.9)

ndre = ndvi - 0.1 + np.random.normal(0, 0.01, (rows, cols))
ndre = np.clip(ndre, 0.1, 0.8)

cigreen = (ndvi / (1 - ndvi + 0.01)) * 0.5

# ---------------------------
# Save as table
# ---------------------------
data = pd.DataFrame({
    "x": xx.flatten(),
    "y": yy.flatten(),
    "NDVI": ndvi.flatten(),
    "NDRE": ndre.flatten(),
    "CIgreen": cigreen.flatten()
})

data.to_csv(output_data, index=False)

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(ndvi, cmap="RdYlGn")
plt.title("NDVI")
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(ndre, cmap="RdYlGn")
plt.title("NDRE")
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(cigreen, cmap="RdYlGn")
plt.title("CIgreen")
plt.colorbar()

plt.tight_layout()
plt.savefig(output_fig, dpi=300)
plt.close()

print("Synthetic UAV data generated.")
print("Saved to:", output_data)
