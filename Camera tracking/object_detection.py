import cv2
import numpy as np
import pyrealsense2 as rs

target_object = input("What are you looking for? \n")

def get_bottle_direction(detections, depth, expected, width, profile):
    min_distance = float("inf")
    direction = ""
    bottle_position = None
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.9:
            label = int(detections[0, 0, i, 1])
            className = classNames[label]

            if className == target_object:
                xmin = int(detections[0, 0, i, 3] * width)
                xmax = int(detections[0, 0, i, 5] * width)
                center_x = (xmin + xmax) / 2

                object_depth = depth[:, xmin:xmax].astype(float) * depth_scale
                dist, _, _, _ = cv2.mean(object_depth)

                if dist < min_distance:
                    min_distance = dist
                    bottle_position = center_x

    if bottle_position is not None:
        third_width = width / 3
        if bottle_position < third_width:
            direction = "Turn left"
        elif bottle_position > 2 * third_width:
            direction = "Turn right"
        else:
            direction = "Go straight ahead"

        print(f"Nearest {target_object} detected at {min_distance:.3f} meters. {direction}")

    return direction

pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
profile = pipe.start(cfg)
align = rs.align(rs.stream.color)
colorizer = rs.colorizer()

prototxt_path = "../MobileNetSSD_deploy.prototxt"
model_path = "../MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

classNames = ("background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair",
              "cow", "diningtable", "dog", "horse",
              "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor")

allowed_objects = {"chair", "person", "bottle"}

while True:
    frameset = pipe.wait_for_frames()
    frameset = align.process(frameset)

    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    if not color_frame or not depth_frame:
        continue

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    height, width = color.shape[:2]
    blob = cv2.dnn.blobFromImage(color, 0.007843, (300, 300), 127.53, False)
    net.setInput(blob, "data")
    detections = net.forward("detection_out")

    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.9:
            label = int(detections[0, 0, i, 1])
            className = classNames[label]

            if className in allowed_objects:
                xmin = int(detections[0, 0, i, 3] * width)
                ymin = int(detections[0, 0, i, 4] * height)
                xmax = int(detections[0, 0, i, 5] * width)
                ymax = int(detections[0, 0, i, 6] * height)

                cv2.rectangle(color, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)
                cv2.putText(color, f"{className} {conf:.2f}", (xmin, ymin - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))

                object_depth = depth[ymin:ymax, xmin:xmax].astype(float)
                depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                object_depth = object_depth * depth_scale
                dist, _, _, _ = cv2.mean(object_depth)

                direction = get_bottle_direction(detections, depth, 300, width, profile)

                if className == target_object and 0.195 <= dist <= 0.205:
                    print("Hello, I reached", target_object)
                    pipe.stop()
                    cv2.destroyAllWindows()
                    exit()

    cv2.imshow("Color Frame", color)
    cv2.imshow("Depth Frame", cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET))

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipe.stop()
cv2.destroyAllWindows()