import sys
sys.path.insert(0, "/content/surveillance")

def test_imports():
    import cv2, supervision as sv, fastapi
    assert True

def test_yolo_load():
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    assert model is not None

def test_supervision_detections():
    import supervision as sv, numpy as np
    boxes = np.array([[100, 100, 200, 200]])
    detections = sv.Detections(xyxy=boxes)
    assert len(detections) == 1

def test_flouter_zone():
    import cv2, numpy as np
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    zone = frame[100:200, 100:200].copy()
    frame[100:200, 100:200] = cv2.GaussianBlur(zone, (51, 51), 0)
    assert not np.array_equal(frame[100:200, 100:200], zone)
