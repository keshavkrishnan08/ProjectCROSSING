#!/usr/bin/env bash
# Download the real object-detection model + sample street footage used by the
# perception demo (scripts/detect_video.py). Kept out of git (large binaries).
set -e
cd "$(dirname "$0")/../assets" 2>/dev/null || { mkdir -p "$(dirname "$0")/../assets"; cd "$(dirname "$0")/../assets"; }

echo "→ COCO class names"
curl -sL -o coco.names "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

echo "→ YOLOv3 ONNX model (~248 MB)"
curl -L -o yolov3.onnx "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/object_detection_segmentation/yolov3/model/yolov3-12.onnx"

echo "→ sample pedestrian footage"
curl -sL -o vtest.avi "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi"

echo "done. Now run:  python -m scripts.detect_video --video assets/vtest.avi --start 40 --count 240"
