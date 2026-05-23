import cv2
from ultralytics import YOLO
import os
import urllib.request
import mediapipe as mp

def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            pass

class HeadDetector:
    def __init__(self, model_path='best.pt'):
        if not os.path.exists(model_path):
            runs_path = 'runs/detect/train/weights/best.pt'
            model_path = runs_path if os.path.exists(runs_path) else 'yolov8n.pt'
        
        self.model = YOLO(model_path)
        
        # Gender model (Adience Dataset by GilLevi)
        self.prototxt = "deploy_gender.prototxt"
        self.caffemodel = "gender_net.caffemodel"
        download_file("https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/deploy.prototxt", self.prototxt)
        download_file("https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_net.caffemodel", self.caffemodel)
        
        self.gender_net = None
        if os.path.exists(self.prototxt) and os.path.exists(self.caffemodel):
            self.gender_net = cv2.dnn.readNetFromCaffe(self.prototxt, self.caffemodel)
            
        self.gender_list = ['MALE', 'FEMALE']
        
        # MediaPipe Hands for Palm detection
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # MediaPipe Face Detection for robust and fast face counting and gender classification
        self.mp_face = mp.solutions.face_detection
        self.face_detector = self.mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.4)
        
        self.head_count = 0
        self.male_count = 0
        self.female_count = 0
        self.palm_detected = False
        self.frame_counter = 0
        self.last_yolo_boxes = []

    def detect(self, frame):
        h, w, _ = frame.shape
        annotated_frame = frame.copy()
        
        self.male_count = 0
        self.female_count = 0
        self.head_count = 0
        
        # Convert frame to RGB once for both face and hand processing to conserve CPU
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. MediaPipe Face Detection (Highly accurate for webcam/classroom settings)
        face_results = self.face_detector.process(rgb_frame)
        
        detected_people = [] # Store coordinates to avoid duplicate YOLO boxes
        
        if face_results.detections:
            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                fw = int(bboxC.width * w)
                fh = int(bboxC.height * h)
                
                # Enforce boundary constraints
                fx = max(0, x)
                fy = max(0, y)
                fx2 = min(w, x + fw)
                fy2 = min(h, y + fh)
                fw = fx2 - fx
                fh = fy2 - fy
                
                if fw > 10 and fh > 10:
                    self.head_count += 1
                    detected_people.append((fx, fy, fx2, fy2))
                    
                    # Crop exact face for perfect gender recognition
                    face_crop = frame[fy:fy2, fx:fx2]
                    gender_str = "UNKNOWN"
                    color = (255, 255, 255) # White default
                    
                    if self.gender_net and face_crop.size > 0:
                        try:
                            blob = cv2.dnn.blobFromImage(face_crop, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                            self.gender_net.setInput(blob)
                            preds = self.gender_net.forward()
                            gender_idx = preds[0].argmax()
                            gender_str = self.gender_list[gender_idx]
                            
                            if gender_str == 'MALE':
                                self.male_count += 1
                                color = (255, 229, 0) # BGR Electric Cyan (#00E5FF)
                            elif gender_str == 'FEMALE':
                                self.female_count += 1
                                color = (108, 65, 255) # BGR Electric Pink (#FF416C)
                            else:
                                color = (255, 255, 255)
                            
                            # Background label rectangle for high-end look
                            label = f"{gender_str}"
                            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(annotated_frame, (fx, fy - text_h - 15), (fx + text_w + 10, fy), color, -1)
                            # Black text on Cyan/Pink/White background for excellent contrast and readability
                            cv2.putText(annotated_frame, label, (fx + 5, fy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            cv2.rectangle(annotated_frame, (fx, fy), (fx2, fy2), color, 2, cv2.LINE_AA)
                        except Exception:
                            cv2.rectangle(annotated_frame, (fx, fy), (fx2, fy2), color, 2, cv2.LINE_AA)
                            # Draw fallback UNKNOWN label
                            (text_w, text_h), baseline = cv2.getTextSize("UNKNOWN", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(annotated_frame, (fx, fy - text_h - 15), (fx + text_w + 10, fy), color, -1)
                            cv2.putText(annotated_frame, "UNKNOWN", (fx + 5, fy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    else:
                        cv2.rectangle(annotated_frame, (fx, fy), (fx2, fy2), color, 2, cv2.LINE_AA)
                        # Draw fallback UNKNOWN label
                        (text_w, text_h), baseline = cv2.getTextSize("UNKNOWN", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(annotated_frame, (fx, fy - text_h - 15), (fx + text_w + 10, fy), color, -1)
                        cv2.putText(annotated_frame, "UNKNOWN", (fx + 5, fy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
        # 2. YOLO Head/Person Detection (For people facing away, or custom head dataset)
        # Optimized with 15-frame skipping to eliminate CPU pegging and deliver a super smooth 30 FPS feed
        if self.frame_counter % 15 == 0 or len(self.last_yolo_boxes) == 0:
            results = self.model(frame, verbose=False, conf=0.3, iou=0.5, classes=[0])[0]
            self.last_yolo_boxes = []
            is_fallback = False
            if hasattr(self.model, 'names') and 0 in self.model.names:
                is_fallback = (self.model.names[0] == 'person')
                
            for box in results.boxes:
                orig_x1, orig_y1, orig_x2, orig_y2 = map(int, box.xyxy[0])
                width = orig_x2 - orig_x1
                height = orig_y2 - orig_y1
                
                if is_fallback:
                    # Fallback: YOLOv8n detected a WHOLE PERSON. Extract head mathematically from the top.
                    head_x1 = max(0, orig_x1 + int(width * 0.15))
                    head_x2 = min(w, orig_x2 - int(width * 0.15))
                    head_y1 = max(0, orig_y1)
                    head_y2 = min(h, orig_y1 + int(height * 0.22))
                else:
                    # Custom model: YOLO detected a HEAD directly.
                    head_x1, head_y1, head_x2, head_y2 = orig_x1, orig_y1, orig_x2, orig_y2
                
                self.last_yolo_boxes.append((head_x1, head_y1, head_x2, head_y2))
                
        self.frame_counter += 1
        
        for (head_x1, head_y1, head_x2, head_y2) in self.last_yolo_boxes:
            # Check if this YOLO head overlaps with an already detected face
            overlap = False
            for (px1, py1, px2, py2) in detected_people:
                ix1 = max(head_x1, px1)
                iy1 = max(head_y1, py1)
                ix2 = min(head_x2, px2)
                iy2 = min(head_y2, py2)
                if ix2 > ix1 and iy2 > iy1:
                    area_intersection = (ix2 - ix1) * (iy2 - iy1)
                    area_face = (px2 - px1) * (py2 - py1)
                    if area_intersection / area_face > 0.4:
                        overlap = True
                        break
                        
            if not overlap:
                self.head_count += 1
                if head_x2 > head_x1 and head_y2 > head_y1:
                    cv2.rectangle(annotated_frame, (head_x1, head_y1), (head_x2, head_y2), (255, 255, 255), 2, cv2.LINE_AA)
                    (text_w, text_h), baseline = cv2.getTextSize("UNKNOWN", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(annotated_frame, (head_x1, head_y1 - text_h - 12), (head_x1 + text_w + 10, head_y1), (255, 255, 255), -1)
                    cv2.putText(annotated_frame, "UNKNOWN", (head_x1 + 5, head_y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                    
        # CLEAN FRAME FOR TELEGRAM (Has identical boxes, but NO hand skeleton)
        clean_frame = annotated_frame.copy()
        
        # 3. Palm Detection (Draw on Live Feed ONLY)
        hand_results = self.hands.process(rgb_frame)
        self.palm_detected = False
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[0]
                middle_mcp = hand_landmarks.landmark[9]
                middle_tip = hand_landmarks.landmark[12]
                
                dist_mcp = ((middle_mcp.x - wrist.x)**2 + (middle_mcp.y - wrist.y)**2)**0.5
                dist_tip = ((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)**0.5
                
                cx, cy = int(middle_mcp.x * w), int(middle_mcp.y * h)
                cv2.circle(annotated_frame, (cx, cy), 30, (0, 255, 150), 3)
                
                if dist_tip > dist_mcp * 1.1:
                    self.palm_detected = True
                    cv2.putText(annotated_frame, "PALM OPEN", (cx - 50, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
                else:
                    cv2.putText(annotated_frame, "HAND DETECTED", (cx - 60, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return annotated_frame, clean_frame, self.head_count, self.male_count, self.female_count, self.palm_detected


def get_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise Exception("Could not open camera")
    return cap
