# Technical Architecture & Engineering Documentation: Local Telemetry Pipeline

**Author:** Harsheet Gaandhi

**Date:** July 2026

**Project Scope:** Local Developer Ergonomics & Operational Analytics

---

## 1. Executive Summary

This document provides the definitive engineering blueprint for the deployment of a local telemetry and context pipeline. The objective is to monitor developer focus, screen attention, and ergonomics by cross-referencing background OS execution cycles with localized computer vision endpoints.

The production architecture is built natively in Python using a deterministic hybrid design: heavy computation, data aggregation, and high-frequency stream filtering occur entirely locally within an isolated virtual environment (`openclaw-env`), while structural insights, temporal profiling, and productivity diagnostics are evaluated via a stateless native LLM runtime.

---

## 2. Environment Architecture & System Prerequisites

To guarantee computational isolation and prevent global dependency poisoning, the pipeline is locked to a specific local Python workspace on the Windows file system.

### Virtual Environment Setup

```cmd
C:\Users\Harsheet Gandhi> python -m venv openclaw-env
C:\Users\Harsheet Gandhi> .\openclaw-env\Scripts\activate
(openclaw-env) C:\Users\Harsheet Gandhi>

```

### Verified Dependency Matrix

The components are highly sensitive to API surface shifts. The pipeline must be locked to the following exact versions. (Note: MediaPipe releases above `0.10.30` strip out the legacy underlying `.solutions` architecture; pinning to `0.10.14` is mathematically required for the framework wrapper bindings to load).

```cmd
pip install opencv-python numpy pandas langchain-google-genai
pip uninstall -y mediapipe
pip install mediapipe==0.10.14

```

---

## 3. Module 1: Computer Vision & Focus Telemetry (`gaze_tracker.py`)

This component leverages a single local camera device endpoint to track physical ocular and skeletal metrics in real time. It utilizes MediaPipe's **Holistic** pipeline to extract facial geometry and postural landmarks concurrently in a single thread, bypassing the overhead of running multi-model pipelines.

### Algorithmic Breakdown

* **Gaze Mechanics:** The script computes normalized geometric vectors mapping the horizontal center of the iris relative to the coordinates of the inner and outer eye corners. A gaze ratio below `0.42` or above `0.58` implies a lateral look away from the display array.
* **Posture Metrics:** It extracts the vertical length between the nose coordinate and the geometric midpoint of the left and right shoulders (`Pose Landmark 11` and `12`). A length falling below a calibrated spatial threshold (`0.15`) indicates an ergonomic slouch (forward neck craning).
* **Frequency Filtering:** To mitigate massive disk I/O bottlenecks and token explosion, the raw frame-by-frame calculations are throttled to write to disk at a fixed interval of exactly $1\text{ Hz}$ (1 row per second).
* **Local Interrupt Service:** Uses a Windows-native `winsound` kernel call to issue an unlatched low-latency audio frequency alert if a distraction state persists past a continuous window of 3 seconds.

### Source Code: `gaze_tracker.py`

```python
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
from datetime import datetime
import winsound

# --- Core Configuration Matrix ---
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
    if not pose_landmarks:
        return "Unknown"
    nose = pose_landmarks.landmark[0]
    l_shoulder = pose_landmarks.landmark[11]
    r_shoulder = pose_landmarks.landmark[12]
    
    mid_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
    neck_length = mid_shoulder_y - nose.y
    
    if neck_length < SLOUCH_THRESHOLD:
        return "Slouching"
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
        
        current_gaze = "Unknown"
        current_posture = "Unknown"
        
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
                    current_gaze, 
                    current_posture, 
                    focus_state
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

```

---

## 4. Module 2: Data Aggregation & Analytical Client Engine (`new_agent_client.py`)

This engine reads the low-frequency background system lifecycle events (`laptop_usage.csv`) and correlates them directly with the high-frequency ergonomic state logs generated by Module 1.

### Data Engineering Design (Hybrid Aggregation)

To achieve low latency and robust execution boundaries, we enforce a separation of concerns:

1. **Local Pre-Aggregation:** The system passes the raw `focus_ergonomics_log.csv` through a local `pandas` transformation matrix. It extracts timestamps, maps the state to a binary integer vector, buckets the tracking metrics into deterministic hour intervals ($0\text{--}23$), and computes an exact percentage of attention loss (`Distraction_Percent`). This downsizes massive logs containing tens of thousands of rows down to a maximum of 24 distinct rows before transmitting payload data.
2. **Stateless LLM Evaluation:** The aggregated text matrices are structured and packaged directly into a LangChain prompt. This avoids brittle agent runtimes and local orchestration sandboxes. The package targets the production `gemini-2.5-flash` model using explicit, secure system instructions.

### Source Code: `new_agent_client.py`

```python
import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

def analyze_combined_telemetry():
    print("[INFO] Initializing Analytics Engine...")
    
    # Absolute path parsing for Windows system tracking architecture
    usage_file = os.path.expanduser("~/laptop_usage.csv")
    focus_file = "focus_ergonomics_log.csv"
    
    usage_data_str = "No laptop usage data available."
    if os.path.exists(usage_file):
        try:
            df_usage = pd.read_csv(usage_file)
            usage_data_str = df_usage.to_string(index=False)
        except Exception as e:
            print(f"[ERROR] Failed to read system telemetry: {e}")
    else:
        print(f"[WARN] System file {usage_file} missing.")

    print("[INFO] Processing local streaming metrics...")
    focus_data_str = "No focus data available."
    if os.path.exists(focus_file):
        try:
            df_focus = pd.read_csv(focus_file)
            df_focus['Timestamp'] = pd.to_datetime(df_focus['Timestamp'])
            df_focus['Is_Distracted'] = (df_focus['Focus_State'] == 'Distracted').astype(int)
            df_focus['Hour'] = df_focus['Timestamp'].dt.hour
            
            # Grouping matrix down into deterministic hours
            hourly_summary = df_focus.groupby('Hour').agg(
                Total_SecondsLogged=('Focus_State', 'count'),
                Distracted_Seconds=('Is_Distracted', 'sum')
            ).reset_index()
            
            hourly_summary['Distraction_Percent'] = (hourly_summary['Distracted_Seconds'] / hourly_summary['Total_SecondsLogged']) * 100
            hourly_summary['Distraction_Percent'] = hourly_summary['Distraction_Percent'].round(1)
            
            focus_data_str = hourly_summary.to_string(index=False)
        except Exception as e:
            print(f"[ERROR] Pandas pipeline transformation breakdown: {e}")
    else:
        print(f"[WARN] Local stream log {focus_file} missing. Run tracking module.")

    if usage_data_str == "No laptop usage data available." and focus_data_str == "No focus data available.":
        print("[CRITICAL] Pipeline empty. Execution halted.")
        return

    # Native Initialization targeting Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )
    
    system_instruction = """
    You are an advanced productivity and ergonomics analyst. You are receiving two sets of telemetry data:
    1. Laptop Usage Logs: Raw system wake/sleep events.
    2. Focus & Ergonomics Logs: Hourly aggregated webcam data showing total seconds tracked and time spent 'Distracted' (looking away or slouching).
    
    Your goal is to correlate these datasets. 
    - Identify the specific hours of the day where the user is most distracted.
    - Calculate the worst distraction percentage.
    - Provide a concise, actionable insight on how they might restructure their work blocks or take breaks based on this specific data.
    
    Format your response cleanly using Markdown headers and bullet points. Do not output raw code.
    """
    
    prompt = f"""
    --- LAPTOP USAGE DATA ---
    {usage_data_str}
    
    --- HOURLY FOCUS SUMMARY ---
    {focus_data_str}
    """
    
    print("\n[INFO] Injecting telemetry matrix into Gemini 2.5 Flash Engine...")
    
    try:
        messages = [
            SystemMessage(content=system_instruction),
            HumanMessage(content=prompt)
        ]
        response = llm.invoke(messages)
        
        print("\n========================================================")
        print("          PRODUCTIVITY & ERGONOMIC ANALYSIS             ")
        print("========================================================\n")
        print(response.content)
        
    except Exception as e:
        print(f"\n[CRITICAL] LLM Network Framework Failure: {e}")

if __name__ == "__main__":
    analyze_combined_telemetry()

```

---

## 5. Deployment Procedures & Execution Sequence

To execute the data processing system without authentication or string escapes breaking the environment, the execution sequence must be followed in this exact order:

### Step 1: Initialize API Environment Authentication

Windows environment variables must be declared **without** literal double quotes. Passing strings with enclosed quotes creates invalid arguments at the gateway handshake layer. Execute the system string registration bare:

```cmd
(openclaw-env) C:\Users\Harsheet Gandhi> set GEMINI_API_KEY=AIzaSyYourCleanActualKeyHere

```

### Step 2: Fire Up the Computer Vision Tracking System

Launch the real-time biometric and focus matrix capturing module. Let this script stream data into your working directory while tasks are processed on screen.

```cmd
(openclaw-env) C:\Users\Harsheet Gandhi> python gaze_tracker.py

```

*(Leave tracking active to generate structural logs. Press `ESC` inside the video context window to close the stream).*

### Step 3: Run the Context Correlation Engine

Run the client pipeline to aggregate the recorded spatial points and invoke the processing layer.

```cmd
(openclaw-env) C:\Users\Harsheet Gandhi> python new_agent_client.py

```

---

## 6. Edge Case Validations & Production Controls

* **Subprocess Sandbox Failures:** Legacy frameworks that run untrusted dynamic scripts in independent runtime sandboxes will drop connection packets on Windows due to environment path resolution collisions. This codebase uses a **deterministic extraction pattern**: all system operations (file checks, matrix mutations, and indexing) are handled natively by your local `pandas` engine, ensuring structural validation before any API payload hits the network layer.
* **Path Separation Discrepancies:** Windows backslash notations (`\`) inside text prompts frequently trip up string literal transformations, resulting in severe `unicodeescape` syntax compilation errors. The pipeline successfully circumvents this by using direct `os.path.expanduser` evaluation blocks combined with automated native file lookups, keeping raw OS file paths entirely out of the text generation contexts.
