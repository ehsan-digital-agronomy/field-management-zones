import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# ---------------------------
# Paths
# ---------------------------
input_data = "../data/synthetic/uav_synthetic.csv"
output_zone_map = "../outputs/figures/management_zones.png"
output_zone_csv = "../outputs/zones/zone_map.csv"

os.makedirs("../outputs/zones", exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
data = pd.read_csv(input_data)

features = data[["NDVI", "NDRE", "CIgreen"]]

# ---------------------------
# K-means clustering
# ---------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
data["Zone"] = kmeans.fit_predict(features)

# Save results
data.to_csv(output_zone_csv, index=False)

# ---------------------------
# Visualization
# ---------------------------
rows = data["y"].max() + 1
cols = data["x"].max() + 1

zone_map = data["Zone"].values.reshape(rows, cols)

plt.figure(figsize=(6,5))
plt.imshow(zone_map, cmap='tab10')
plt.title("Management Zones (K-means, k=3)")
plt.colorbar(label="Zone")
plt.tight_layout()
plt.savefig(output_zone_map, dpi=300)
plt.close()

print("Management zones created.")
