from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2
import io
import os
import requests
import asyncio
import json
from head_detector import HeadDetector, get_camera
import threading
import time
from collections import deque
from datetime import datetime
import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

STITCH_MCP_CONFIG = {
    "serverUrl": "https://stitch.googleapis.com/mcp",
    "enabled": True
}

app = FastAPI()
detector = HeadDetector()

camera_lock = threading.Lock()
current_frame = None
head_count = 0
male_count = 0
female_count = 0
palm_triggered = False
palm_has_been_triggered = False # State variable to lock trigger until manually reset from UI
last_palm_seen_time = 0 # Timestamp when palm was last detected for temporal debounce

last_viewed_time = 0
palm_start_time = 0
palm_cooldown = 0
auto_report_cooldown = 0

video_buffer = deque(maxlen=150)
last_count = 0
stable_frames = 0
is_recording = False

# Dedicated background thread to clear camera buffer
class CameraBuffer:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.active = True # ALWAYS ACTIVE FOR INSTANT START!
        threading.Thread(target=self.update, daemon=True).start()
        
    def set_active(self, active_state):
        # Keep active permanently
        pass
            
    def update(self):
        while self.running:
            with self.lock:
                should_be_active = self.active
                
            if should_be_active:
                if self.cap is None:
                    try:
                        self.cap = get_camera()
                    except Exception:
                        self.cap = None
                        
                if self.cap is not None:
                    try:
                        success, f = self.cap.read()
                        if success:
                            with self.lock:
                                self.frame = f
                        time.sleep(0.01)
                    except Exception:
                        time.sleep(0.01)
                else:
                    time.sleep(0.5)
            else:
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
                with self.lock:
                    self.frame = None
                time.sleep(0.2)

    def get_frame(self):
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

cam_buffer = CameraBuffer()

def process_and_send_auto(frames_to_process, count):
    global is_recording, tg_token, tg_chat_id
    if not tg_token or not tg_chat_id:
        is_recording = False
        return
        
    filename = f"auto_report_{int(time.time())}.mp4"
    try:
        if len(frames_to_process) > 0:
            h, w, _ = frames_to_process[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
            for f in frames_to_process:
                out.write(f)
            out.release()
            
            url = f"https://api.telegram.org/bot{tg_token}/sendVideo"
            caption = f"🤖 VISIONCORE AUTO-ALERT 🤖\n\nStable Target Count: {count}\nStatus: Tracking Active"
            with open(filename, 'rb') as video:
                files = {'video': video}
                data = {'chat_id': tg_chat_id, 'caption': caption}
                requests.post(url, data=data, files=files)
    except Exception as e:
        print("Auto broadcast failed:", e)
    finally:
        if os.path.exists(filename):
            os.remove(filename)
        is_recording = False

def send_attendance_report(frame, count, males, females):
    global tg_token, tg_chat_id
    if not tg_token or not tg_chat_id:
        return
        
    filename = f"attendance_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    
    try:
        url = f"https://api.telegram.org/bot{tg_token}/sendPhoto"
        
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        day_of_week = now.strftime("%A")
        
        caption = (
            f"📝 ATTENDANCE REPORT 📝\n\n"
            f"Total Strength: {count}\n"
            f"No of Male Present: {males}\n"
            f"No of Female Present: {females}\n"
            f"Time Taken: {time_str}\n"
            f"Day of Weekend: {day_of_week}\n\n"
            f"Dataset Used: Adience Benchmark Gender Dataset\n"
            f"Creator: Aariz\n"
            f"System: Robocoupler VisionCore"
        )
        
        with open(filename, 'rb') as photo:
            files = {'photo': photo}
            data = {'chat_id': tg_chat_id, 'caption': caption}
            requests.post(url, data=data, files=files)
    except Exception as e:
        print("Attendance send failed:", e)
    finally:
        if os.path.exists(filename):
            os.remove(filename)

def video_capture_loop():
    global current_frame, head_count, male_count, female_count, palm_triggered
    global palm_start_time, palm_cooldown, video_buffer, last_count, stable_frames, is_recording
    global auto_report_cooldown, palm_has_been_triggered, last_palm_seen_time
    
    while True:
        try:
            # Persistent camera loop for instant page load and zero connection delay
            frame = cam_buffer.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            
            annotated_frame, clean_frame, h_count, m_count, f_count, palm_det = detector.detect(frame)
            video_buffer.append(clean_frame)
            
            with camera_lock:
                current_frame = annotated_frame
                head_count = h_count
                male_count = m_count
                female_count = f_count
                
                # Manual Palm Trigger Logic (Sends Photo & 5-Second Video Buffer)
                # Trigger exactly ONCE. Lock trigger until manually reset from UI!
                if palm_det:
                    last_palm_seen_time = time.time() # Update last seen timestamp
                    if not palm_has_been_triggered:
                        if palm_start_time == 0:
                            palm_start_time = time.time()
                        elif time.time() - palm_start_time >= 0.5: # Half second hold
                            palm_cooldown = time.time()
                            palm_triggered = True
                            palm_has_been_triggered = True # Lock trigger
                            
                            # 1. Send Attendance Photo (Clean of Hand Skeleton)
                            threading.Thread(target=send_attendance_report, args=(clean_frame.copy(), h_count, m_count, f_count)).start()
                            
                            # 2. Send 5-Second Video Buffer
                            if not is_recording:
                                is_recording = True
                                frames_to_process = list(video_buffer)
                                threading.Thread(target=process_and_send_auto, args=(frames_to_process, h_count)).start()
                else:
                    palm_start_time = 0
                    palm_triggered = False
            
            time.sleep(0.01)
        except Exception as e:
            print(f"CRASH IN LOOP: {e}")
            time.sleep(1)

thread = threading.Thread(target=video_capture_loop, daemon=True)
thread.start()

@app.get("/video_feed")
async def video_feed():
    async def generate():
        global last_viewed_time
        try:
            while True:
                last_viewed_time = time.time()
                frame_to_yield = None
                with camera_lock:
                    if current_frame is not None:
                        ret, buffer = cv2.imencode('.jpg', current_frame)
                        frame_to_yield = buffer.tobytes()
                
                if frame_to_yield is None:
                    await asyncio.sleep(0.05)
                    continue
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_to_yield + b'\r\n')
                await asyncio.sleep(0.03)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stats")
async def get_stats():
    with camera_lock:
        return {
            "head_count": head_count,
            "male_count": male_count,
            "female_count": female_count,
            "palm_triggered": palm_triggered,
            "palm_locked": palm_has_been_triggered
        }

@app.post("/api/attendance/reset")
async def reset_attendance():
    global palm_has_been_triggered, palm_triggered, palm_start_time
    with camera_lock:
        palm_has_been_triggered = False
        palm_triggered = False
        palm_start_time = 0
    return {"success": True}

class TelegramConfig(BaseModel):
    token: str
    chat_id: str = ""

CONFIG_FILE = "telegram_config.json"
tg_token = ""
tg_chat_id = ""

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r') as f:
            conf = json.load(f)
            tg_token = conf.get("token", "")
            tg_chat_id = conf.get("chat_id", "")
    except Exception:
        pass

@app.get("/api/telegram/status")
async def telegram_status():
    return {"connected": bool(tg_token and tg_chat_id), "token": tg_token, "chat_id": tg_chat_id}

@app.post("/api/telegram/connect")
async def connect_telegram(config: TelegramConfig):
    global tg_token, tg_chat_id
    tg_token = config.token
    tg_chat_id = config.chat_id
    
    if not tg_chat_id:
        try:
            res = requests.get(f"https://api.telegram.org/bot{tg_token}/getUpdates").json()
            if res.get("ok") and res.get("result"):
                tg_chat_id = str(res["result"][-1]["message"]["chat"]["id"])
            else:
                return {"success": False, "error": "No chat found. Please send a message to your bot first."}
        except Exception as e:
            return {"success": False, "error": str(e)}
            
    with open(CONFIG_FILE, 'w') as f:
        json.dump({"token": tg_token, "chat_id": tg_chat_id}, f)
        
    return {"success": True, "chat_id": tg_chat_id}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    local_ip = get_local_ip()
    print("\n" + "="*60)
    print("🌐 VISIONCORE REAL-TIME INTELLIGENCE DASHBOARD ACTIVE 🌐")
    print("="*60)
    print(f"👉 Local Access:   http://localhost:8000")
    print(f"👉 Network Access: http://{local_ip}:8000")
    print("="*60)
    print("📢 Tip: Open the Network Access URL from your mobile phone or another PC")
    print("📢 on the same Wi-Fi network to monitor the classroom attendance!")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
