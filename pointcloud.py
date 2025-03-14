import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import matplotlib
import os
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

# Erstelle eine Pipeline und konfiguriere die Streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Starte die Pipeline
pipeline.start(config)

# Hole die intrinsischen Parameter der Kamera
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale: {depth_scale}")

depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Stelle sicher, dass der Visualizer initialisiert wird
vis = None

try:
    # Warte auf Frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        raise ValueError("Konnte keine Frames von der Kamera abrufen.")

    # Konvertiere Tiefendaten in ein NumPy-Array
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Visualisiere das Tiefenbild zur Diagnose
    plt.imshow(np.clip(depth_image, 0, 5000), cmap='jet')
    plt.colorbar()
    plt.title("Depth Image")
    plt.show()

    # Hole die Bilddimensionen
    height, width = depth_image.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.flatten()
    y = y.flatten()

    # Hole die Tiefenwerte und skaliere sie
    depth = depth_image.flatten() * depth_scale  # Umwandlung von mm in Meter

    # Überprüfung der Tiefenwerte vor dem Filtern
    print(f"Min Depth: {np.min(depth)} m, Max Depth: {np.max(depth)} m")

    # Berechne die 3D-Koordinaten (X, Y, Z) mit den intrinsischen Parametern
    fx, fy = depth_intrinsics.fx, depth_intrinsics.fy
    ppx, ppy = depth_intrinsics.ppx, depth_intrinsics.ppy

    X = (x - ppx) * depth / fx
    Y = (y - ppy) * depth / fy
    Z = depth

    # Filter für valide Tiefenwerte (zwischen 0.3m und 5m)
    valid_depth = (Z >= 0) & (Z < 3.0)

    # Überprüfung nach Filterung
    filtered_depth = depth[valid_depth]
    if len(filtered_depth) >= 0:
        print(f"Min gültige Tiefe: {np.min(filtered_depth)} m, Max gültige Tiefe: {np.max(filtered_depth)} m")
        print(f"Anzahl gültiger Pixel: {len(filtered_depth)}")
    else:
        print("Keine gültigen Tiefenwerte gefunden!")
        raise ValueError("Keine validen Punkte für die Punktwolke.")

    X, Y, Z = X[valid_depth], Y[valid_depth], Z[valid_depth]
    colors = color_image.reshape(-1, 3)[valid_depth] / 255.0  # Normalisierte Farben

    # Erstelle die Punktwolke
    points = np.vstack((X, Y, Z)).T
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Voxel Grid Downsampling
    voxel_size = 0.01  # 1 cm Auflösung
    downsampled_pcd = point_cloud.voxel_down_sample(voxel_size)

    # Entferne Ausreißer mit Statistical Outlier Removal (SOR)
    cl, ind = downsampled_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    filtered_pcd = downsampled_pcd.select_by_index(ind)

    # Entferne isolierte Punkte mit Radius Outlier Removal (Radius-Based Filter)
    cl, ind = filtered_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    filtered_pcd = filtered_pcd.select_by_index(ind)

    # Optional: Glättung durch weiteres Downsampling
    filtered_pcd = filtered_pcd.voxel_down_sample(voxel_size=0.02)

    # Zeige die bereinigte Punktwolke
    o3d.visualization.draw_geometries([filtered_pcd])

    # Erstelle ein Koordinatensystem mit einer Länge von 0.5 Metern
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    # Zeige die Punktwolke mit dem Koordinatensystem
    o3d.visualization.draw_geometries([filtered_pcd, coordinate_frame])

    # Visualisierung
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(downsampled_pcd)
    vis.get_render_option().point_size = 3.0
    vis.run()

    points = np.asarray(filtered_pcd.points)
    df = pd.DataFrame(points, columns=["X", "Y", "Z"])
    df.to_csv("filtered_pointcloud.csv", index=False)
    print("Punktwolke als CSV gespeichert!")

finally:
    # Stoppe die Pipeline und schließe das Fenster
    pipeline.stop()
    if vis:
        vis.destroy_window()