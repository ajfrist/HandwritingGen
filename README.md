# HandwritingGen

HandwritingGen is an application for synthesizing handwriting from a user's training input data. The app lets you train a model on user-provided handwriting samples and then generate new handwriting images or stroke sequences in the learned style. The project separates concerns into a Python backend (model training / inference) and a JavaScript frontend (UI for uploading samples, starting training, previewing synthesized handwriting).

---

## Key Features

- Train a handwriting synthesis model on user-provided samples
- Serve model inference via an HTTP API
- Web UI for uploading training data, monitoring training, and generating samples

---

## Target Users

- Programmers and Data Science researchers and students who want a quick pipeline for handwriting synthesis experiments
- Developers building handwriting-style features (font generation, stylized text)
- Hobbyists interested in generative handwriting models

---

## Tech Stack

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

## Environment Variables

You must configure an environment variable for the frontend to connect to the proper IP address. Create the file `.env` in the `frontend/` directory, and save the IP address on which you will run the server. This could be `localhost` or your local IP in the form of `EXPO_PUBLIC_API_URL=http://<IPaddress>:<port>`.

Example frontend `.env` with default, local frontend connection configuration:
```
EXPO_PUBLIC_API_URL=http://localhost:5000
```

**Important**: Replace `localhost` with the IP address of the machine running your backend if you want to access it from other devices on the same LAN. Passing `0.0.0.0` to the backend host exposes it to the network (uvicorn / FastAPI CLI uses `--host 0.0.0.0`), but API clients need the concrete IP (not `0.0.0.0`) to call from outside. To get yours, enter the `ipconfig` command into the terminal and look for IPv4 Address.

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
# from repository root, with venv active
python -m pip install --upgrade pip
pip install -r python/requirements.txt
```

3. Run the backend (FastAPI / Uvicorn example)
```bash
# from repository root
cd python
# ensure venv is active
python -m uvicorn api_server:app --port 5000 --reload
```
  To expose backend to devices on the LAN network, insead run:
```bash
python -m uvicorn api_server:app --host 0.0.0.0 --port 5000 --reload
```

### 3) Frontend (Node) setup

1. Install dependencies
```bash
# from repository root
cd frontend
npm install
```

2. Configure frontend `.env` (see above [Environment Variables](#environment-variables)):

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

