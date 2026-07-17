import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime
import winsound

# Configuration
LOG_FILE = "focus_ergonomics_log.csv"
DISTRACTION_THRESHOLD_SEC = 3.0
ALERT_COOLDOWN_SEC = 5.0
SLOUCH_THRESHOLD = 0.15 

def estimate_gaze(iris_center, inner_corner, outer_corner):
    eye_width = np.linalg.norm(outer_corner - inner_corner)
    iris_dist = np.linalg.norm(iris_center - inner_corner)
    ratio = iris_dist / eye_width if eye_width > 0 else 0.5
    
    if ratio < 0.42: return "Looking Right"
    elif ratio > 0.58: return "Looking Left"
    return "Center"

def analyze_posture(pose_landmarks):
    if not pose_landmarks: return "Unknown"
    nose = pose_landmarks.landmark[0]
    l_shoulder = pose_landmarks.landmark[11]
    r_shoulder = pose_landmarks.landmark[12]
    
    mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
    neck_length = mid_shoulder_y - nose.y
    
    if neck_length < SLOUCH_THRESHOLD: return "Slouching"
    return "Good"

def main():
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        refine_face_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    
    cap = cv2.VideoCapture(0)
    distracted_start_time = None
    last_alert_time = 0
    last_log_time = 0
    
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Timestamp", "Gaze", "Posture", "Focus_State"])

    print(f"[INFO] Tracking active. Target log: {LOG_FILE}. Esc to terminate.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        
        current_gaze, current_posture = "Unknown", "Unknown"
        
        if results.face_landmarks:
            landmarks = results.face_landmarks.landmark
            def get_coord(idx): return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
            r_gaze = estimate_gaze(get_coord(468), get_coord(133), get_coord(33))
            l_gaze = estimate_gaze(get_coord(473), get_coord(362), get_coord(263))
            current_gaze = r_gaze if r_gaze == l_gaze else "Center"

        if results.pose_landmarks:
            current_posture = analyze_posture(results.pose_landmarks)
            
        is_focused = (current_gaze == "Center" and current_posture == "Good")
        focus_state = "Focused"
        current_time = time.time()
        
        if not is_focused:
            if distracted_start_time is None:
                distracted_start_time = current_time
            elif (current_time - distracted_start_time) > DISTRACTION_THRESHOLD_SEC:
                focus_state = "Distracted"
                if (current_time - last_alert_time) > ALERT_COOLDOWN_SEC:
                    winsound.Beep(1000, 500)
                    last_alert_time = current_time
        else:
            distracted_start_time = None
            
        if (current_time - last_log_time) >= 1.0:
            with open(LOG_FILE, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                    current_gaze, current_posture, focus_state
                ])
            last_log_time = current_time

        color = (0, 255, 0) if focus_state == "Focused" else (0, 0, 255)
        cv2.putText(frame, f"Gaze: {current_gaze}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Posture: {current_posture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"State: {focus_state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow('Ergonomic & Focus Tracker', frame)
        if cv2.waitKey(5) & 0xFF == 27: break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
