from ultralytics import YOLO
import cv2
import torch
import numpy as np
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

# ✅ Ensure GPU is being used
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# ✅ Load YOLO model
model = YOLO("../YOLO-Weights/yolov8s.pt").to(device)

# ✅ Initialize DeepSORT Tracker
cfg_deep = get_config()
cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

deepsort = DeepSort(
    cfg_deep.DEEPSORT.REID_CKPT,
    max_dist=cfg_deep.DEEPSORT.MAX_DIST,
    min_confidence=0.3,
    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=10,
    n_init=3,
    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
    use_cuda=torch.cuda.is_available()
)

# ✅ Motion Tracking Storage
object_counters = {}
track_history = {}
max_trail_length = 15
frame_skip = 2
frame_count = 0


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    """ Resize image with aspect ratio intact and pad. """
    shape = image.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw // 2, dh // 2

    image_resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    image_padded = cv2.copyMakeBorder(image_resized, dh, dh, dw, dw, cv2.BORDER_CONSTANT, value=color)

    return image_padded, ratio, dw, dh


def video_detection(path_x):
    global object_counters, track_history, frame_count
    # ✅ Reset tracking for each video
    object_counters = {}
    track_history = {}
    frame_count = 0

    cap = cv2.VideoCapture(path_x)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {path_x}")
        return

    while cap.isOpened():
        success, img = cap.read()
        if not success or img is None:
            print("❌ Error: Failed to read frame.")
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        orig_h, orig_w = img.shape[:2]

        # ✅ Apply Letterboxing for YOLO input
        img_resized, ratio, dw, dh = letterbox(img, new_shape=(640, 640))

        # ✅ Normalize Image Input
        img_resized = img_resized.astype(np.float32) / 255.0

        # ✅ Convert image to tensor (B, C, H, W)
        img_gpu = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0).to(device)

        # ✅ Run YOLO inference
        results = model.predict(img_gpu, conf=0.3, imgsz=640, device=device)

        bbox_xywh = []
        confs = []
        class_ids = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]

                # ✅ Scale bounding boxes back to original size
                x1 = int((x1 - dw) / ratio)
                y1 = int((y1 - dh) / ratio)
                x2 = int((x2 - dw) / ratio)
                y2 = int((y2 - dh) / ratio)

                w, h = x2 - x1, y2 - y1
                bbox_xywh.append([x1, y1, w, h])
                confs.append(box.conf[0].item())  # ✅ Store confidence for tracking
                class_ids.append(int(box.cls[0]))

        if len(bbox_xywh) > 0:
            xywhs = torch.Tensor(bbox_xywh).to(device)
            confss = torch.Tensor(confs).to(device)

            # ✅ Convert CUDA tensors to NumPy before passing to DeepSORT
            outputs = deepsort.update(
                xywhs.cpu().numpy(),
                confss.cpu().numpy(),
                class_ids,
                img
            )

            for i, output in enumerate(outputs):
                x1, y1, x2, y2, track_id, class_id = output
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                class_names = model.names
                class_name = class_names.get(class_id, "Unknown")

                if class_name not in object_counters:
                    object_counters[class_name] = {}
                if track_id not in object_counters[class_name]:
                    object_counters[class_name][track_id] = len(object_counters[class_name]) + 1

                object_num = object_counters[class_name][track_id]

                # ✅ Correctly Assign Confidence Score
                confidence = confs[i] if i < len(confs) else 0.0

                # ✅ Restore Motion Tracking
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if track_id not in track_history:
                    track_history[track_id] = deque(maxlen=max_trail_length)
                track_history[track_id].appendleft((center_x, center_y))

                # ✅ Apply Blurring Effect
                obj_region = img[y1:y2, x1:x2]
                if obj_region.shape[0] > 0 and obj_region.shape[1] > 0:
                    blurred_obj = cv2.GaussianBlur(obj_region, (35, 35), 30)
                    img[y1:y2, x1:x2] = blurred_obj

                # ✅ Display Label with Correct Confidence Score
                label = f'{class_name} {object_num} ({confidence:.2f})'
                cv2.putText(img, label, (x1, y1 - 10), 0, 1, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ✅ Draw Motion Trails
                for j in range(1, len(track_history[track_id])):
                    if track_history[track_id][j - 1] is None or track_history[track_id][j] is None:
                        continue
                    thickness = int(np.sqrt(max_trail_length / float(j + j)) * 1.5)
                    cv2.line(img, track_history[track_id][j - 1], track_history[track_id][j], (0, 255, 255), thickness)

        yield img

    cap.release()
    cv2.destroyAllWindows()
