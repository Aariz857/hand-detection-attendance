# VisionCore AI - Head Count System

A premium, offline-capable head recognition and counting system using YOLOv8.

## Features
- **Real-time Detection**: Uses your local camera to detect and count heads.
- **Premium Dashboard**: Sleek, futuristic UI with glassmorphism and real-time updates.
- **Offline First**: Runs entirely on your local machine.
- **Easy Training**: Includes a script to train on your custom head dataset.

## How to Run
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train the Model** (Optional - if you want to use your dataset):
    ```bash
    python train_model.py
    ```
    *Note: This will create `runs/detect/train/weights/best.pt`.*

3.  **Start the Application**:
    ```bash
    python app.py
    ```
    OR simply double-click `run.bat`.

4.  **Access the Dashboard**:
    Open your browser and go to `http://localhost:8000`.

## Directory Structure
- `app.py`: FastAPI backend server.
- `head_detector.py`: YOLOv8 inference logic.
- `train_model.py`: Training script for the provided dataset.
- `static/`: Frontend files (HTML, CSS, JS).
- `data.yaml`: Dataset configuration.

## Requirements
- Python 3.8+
- Webcam
- Internet (only for initial package installation and weight download)

## Telegram Bot Integration
To receive automatic headcount reports, stability alerts, and open-palm snapshot captures directly on your mobile device:
1. Message `@BotFather` on Telegram and send `/newbot` to create your bot. Copy the generated **HTTP API Token**.
2. Create a file named `telegram_config.json` in the root directory (you can copy and rename `telegram_config.json.template` as a base).
3. Paste your token in `telegram_config.json`:
   ```json
   {
     "token": "YOUR_TELEGRAM_BOT_TOKEN",
     "chat_id": ""
   }
   ```
4. Start your bot by sending it a message (`/start` or any text) in the Telegram app.
5. Launch the application (`run.bat` or `python app.py`). The server will automatically query Telegram to fetch your chat ID and complete the setup!

