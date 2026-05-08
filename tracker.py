import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 Nano model
# The 'n' stands for nano. It's the fastest version for real-time video.
# (It will auto-download a tiny ~6MB yolov8n.pt file the first time you run this)
model = YOLO("yolov8n.pt")

# 2. Define the Restricted Zone Polygon 
# These are [X, Y] pixel coordinates on your camera feed. 
# We define 4 points to make a trapezoid, but you can change these later.
zone_pts = np.array([[150, 150], [500, 150], [600, 450], [50, 450]], np.int32)
zone_pts = zone_pts.reshape((-1, 1, 2))

# 3. Start the Webcam (0 is usually the default laptop webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Pipeline started! Press 'q' on the video window to quit.")

while True:
    # Read the current frame from the camera
    success, frame = cap.read()
    if not success:
        break

    # Default zone color is Green (Safe)
    zone_color = (0, 255, 0)
    alert_triggered = False

    # 4. Run YOLOv8 inference
    # stream=True keeps it fast, verbose=False stops it from flooding your terminal
    results = model(frame, stream=True, verbose=False)

    # 5. Process the detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Check the class ID (0 is 'person' in the standard COCO dataset)
            cls_id = int(box.cls[0])
            
            if cls_id == 0:
                # Get bounding box coordinates: Top-Left (x1, y1) and Bottom-Right (x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Logic Engine: Where is the person standing?
                # We calculate the bottom-center of the box (representing their feet)
                center_x = int((x1 + x2) / 2)
                bottom_y = y2
                
                # 6. Check if the feet are inside the Polygon
                # pointPolygonTest returns a positive number if inside, negative if outside
                is_inside = cv2.pointPolygonTest(zone_pts, (center_x, bottom_y), False)
                
                if is_inside >= 0:
                    alert_triggered = True
                    # Draw Red Box & Dot for violators
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(frame, (center_x, bottom_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "VIOLATION", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    print("WARNING: Person detected in Restricted Zone!")
                else:
                    # Draw Green Box & Dot for people safely outside the zone
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, bottom_y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, "Safe", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 7. Dynamic UI: Turn the polygon Red if there is a breach
    if alert_triggered:
        zone_color = (0, 0, 255) 
        cv2.putText(frame, "ALERT: ZONE BREACH", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Draw the boundary lines of the zone
    cv2.polylines(frame, [zone_pts], isClosed=True, color=zone_color, thickness=3)

    # 8. Display the live video feedpython tracker.py
    cv2.imshow("Biohazard Tracker - Phase 1 Pipeline", frame)

    # 9. Quit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up your system's resources when done
cap.release()
cv2.destroyAllWindows()