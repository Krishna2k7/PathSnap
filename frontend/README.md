# PathSnap – Evaluator Frontend (Laptop 2 Client)

A modern, high-performance **React 18 + Vite** evaluation interface for **PathSnap: Distributed Map-Matching & Route Reconstruction**.

Built for the two-laptop hackathon architecture where the Machine Learning inference and HMM/Viterbi sequence reconstruction run on **Laptop 1**, while the interactive visualization and evaluator controls run on **Laptop 2**.

---

## 🚀 Key Features

- **Dynamic Trajectory Selection:** Dynamically queries available evaluation trips from the backend registry.
- **Dual-Layer Leaflet Geographical Map:**
  - 🔵 **Blue Markers:** Raw, noisy GPS observation fixes with coordinates and sequence IDs.
  - 🔴 **Red Polyline:** Viterbi-decoded road-network reconstructed route.
  - 🎚️ **Interactive Layer Controls:** Toggle raw GPS fixes and reconstructed path on demand.
- **Server-Side Evaluation Dashboard:**
  - Route Continuity (HMM network connectivity)
  - Observation-level Top-1 Accuracy (Pointwise candidate classification)
  - HMM Path Accuracy
  - Candidate Macro F1-score
  - Candidate Miss Rate (%)
  - Probability Calibration (Log Loss & Brier Score)
  - Inference Latency (ms)
- **Strict Academic Integrity & Security:**
  - Evaluator ground-truth paths are **never** transmitted to the browser.
  - Zero hardcoded routes or fabricated metrics.
  - All predictions are dynamically computed by the server's `HistGradientBoostingClassifier` and HMM decoder.

---

## 🛠️ Tech Stack

- **Framework:** React 18 + Vite
- **Styling:** Tailwind CSS + Lucide React icons
- **Mapping:** Leaflet.js with CartoDB Voyager tiles
- **Networking:** Local REST API client with timeout and connection-resilience handlers

---

## 💻 Two-Laptop Setup & Execution Guide

### Step 1: Start the Backend on Laptop 1

1. Open a terminal on **Laptop 1** in the repository root:
   ```bash
   # Run the production API server
   python api/server.py
   ```
2. Find Laptop 1's Local IP address:
   - On Windows: `ipconfig` (look for *IPv4 Address*, e.g., `192.168.1.15`)
   - On Linux/macOS: `ip addr` or `ifconfig`

---

### Step 2: Configure & Start Frontend on Laptop 2

1. Clone or open the repository on **Laptop 2** and navigate to the `frontend/` directory:
   ```bash
   cd frontend
   ```

2. Copy the environment template and point to Laptop 1's IP:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```ini
   VITE_API_BASE_URL=http://192.168.1.15:8000
   ```
   *(Replace `192.168.1.15` with Laptop 1's actual LAN IP)*

3. Install dependencies:
   ```bash
   npm install
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```
   Open the displayed URL (e.g. `http://localhost:5173/`) in your browser.

---

### Production Build (Optional)

To compile an optimized static production bundle:
```bash
npm run build
npm run preview
```
