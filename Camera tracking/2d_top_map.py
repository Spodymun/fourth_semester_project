import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

# Sicherstellen, dass Matplotlib im interaktiven Modus läuft
matplotlib.use("TkAgg")


# CSV-Datei laden und Punktwolke einlesen
def load_point_cloud_from_csv(file_path):
    df = pd.read_csv(file_path)
    return df[['X', 'Y', 'Z']].values


# Höhenkarte erstellen
def create_heightmap(points, resolution=0.05):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Min/Max für Gitter bestimmen
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_bins = np.arange(x_min, x_max, resolution)
    y_bins = np.arange(y_min, y_max, resolution)

    heightmap = np.zeros((len(y_bins), len(x_bins)))

    for i in range(len(points)):
        x_idx = np.searchsorted(x_bins, x[i]) - 1
        y_idx = np.searchsorted(y_bins, y[i]) - 1

        if 0 <= x_idx < len(x_bins) and 0 <= y_idx < len(y_bins):
            heightmap[y_idx, x_idx] = max(heightmap[y_idx, x_idx], z[i])

    return heightmap


# Höhenkarte anzeigen
def plot_heightmap(heightmap):
    plt.ion()  # Interaktiver Modus aktivieren
    plt.imshow(heightmap, cmap="terrain", origin="lower")
    plt.colorbar()
    plt.title("Heightmap")
    plt.draw()
    plt.pause(0.1)  # Kurze Pause für Anzeige
    plt.ioff()  # Interaktiver Modus deaktivieren
    plt.show()


# Hauptprogramm
if __name__ == "__main__":
    csv_file = "pointclouds/filtered_pointcloud.csv"  # Ersetze mit deinem Dateinamen
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Datei {csv_file} nicht gefunden!")

    point_cloud = load_point_cloud_from_csv(csv_file)
    heightmap = create_heightmap(point_cloud, resolution=0.02)
    plot_heightmap(heightmap)