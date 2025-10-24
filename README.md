# HandwritingGen

HandwritingGen is an application for synthesizing handwriting from a user's training input data. The app lets you train a model on user-provided handwriting samples and then generate new handwriting images or stroke sequences in the learned style. The project separates concerns into a Python backend (model training / inference) and a JavaScript frontend (UI for uploading samples, starting training, previewing synthesized handwriting).

---

## Key features

- Train a handwriting synthesis model on user-provided samples
- Serve model inference via an HTTP API
- Web UI for uploading training data, monitoring training, and generating samples

---

## Target users

- Programmers and Data Science researchers and students who want a quick pipeline for handwriting synthesis experiments
- Developers building handwriting-style features (font generation, stylized text)
- Hobbyists interested in generative handwriting models

---

## Tech stack

- Backend: Python (FastAPI recommended) for model training, inference, and REST API with Uvicorn
- Model code: Numpy to store and cross-compute across data
- Frontend: React (create-react-app / Vite) with Babel
- Development tooling: npm / npx for frontend utilities, Python venv for backend isolation

---

## Prerequisites

- Python 3 (3.13 recommended)
- pip
- Node.js 16+ and npm (npx is included with npm)
- A local IP address for cross-device testing (if you want to access the backend from another machine / phone)
- (Optional) Expo Go mobile app (for running app on phone or device other than web)

---

## Environment variables

You must configure an environment variable for the frontend to connect to the proper IP address. Create `.env` files in the `frontend/` directory, and save your LAN IP address. 

Example frontend `.env` (frontend/.env):
```
API_URL=http://10.110.29.137:5000

```

Important: Replace `192.168.1.10` with the IP address of the machine running your backend if you want to access it from other devices on the same LAN. Using `0.0.0.0` as the backend host exposes it to the network (uvicorn / FastAPI CLI uses `--host 0.0.0.0`), but API clients need the concrete IP (not `0.0.0.0`) to call from outside. To get yours, enter the `ipconfig` command into the terminal and look for IPv4 Address.

---

## Setup (step-by-step)

Below are typical steps to prepare and run both backend and frontend locally.

### 1) Clone the repo
```bash
git clone https://github.com/ajfrist/HandwritingGen.git
cd HandwritingGen
```

### 2) Backend (Python) setup

1. Create and activate a virtual environment

- Linux / macOS:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- Windows (PowerShell):
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

2. Install Python packages
```bash
cd backend
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the backend (FastAPI / Uvicorn example)
```bash
# from repository root
cd python
# ensure venv is active
python -m uvicorn app.api_server:app --reload --host 0.0.0.0 --port 5000
```

### 3) Frontend (Node) setup

1. Install dependencies and create frontend env
```bash
# from repository root
cd frontend
npm install
```

2. Configure frontend `.env` (inside `frontend/.env`):
```
API_URL=http://<BACKEND_IP>:5000/
```
Replace `<BACKEND_IP>` with your machine IP (e.g., `192.168.1.10`) or `localhost` if accessing from same machine.

3. Run the frontend (development)
```bash
npx expo start
```
This will display directions in the terminal for running the application on the web or in emulator. To run on a physical mobile device, scan the QR code with the Expo Go app. 

### 4) Mobile Application Connection

1. Download Expo Go
- Install Expo Go from your mobile app store.

2. Scan QR Code
- Open the Expo Go app and scan the QR code generated in the terminal.
- This will build the application for your device and allow you to use it on external hardware. 

