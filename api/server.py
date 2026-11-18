"""
PathSnap Server Launcher (api/server.py)
========================================
Production launcher for the FastAPI backend service on Laptop 1.
Serves the API and interactive web evaluator UI on 0.0.0.0:8000.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from api.main import app, get_evaluation_base_dir

def run():
    host = os.environ.get("PATHSNAP_HOST", "0.0.0.0")
    port = int(os.environ.get("PATHSNAP_PORT", "8000"))
    eval_dir = get_evaluation_base_dir()

    print("=" * 75)
    print("        PATHSNAP PRODUCTION TWO-LAPTOP EVALUATOR SERVICE")
    print("=" * 75)
    print(f" [Host]:               http://{host}:{port}")
    print(f" [Local Browser UI]:   http://localhost:{port}/")
    print(f" [Laptop 2 LAN UI]:    http://<LAPTOP_1_IP>:{port}/")
    print(f" [Health Check]:       http://localhost:{port}/health")
    print(f" [Trip Discovery]:     http://localhost:{port}/trips")
    print(f" [Active Partition]:   {eval_dir}")
    print("=" * 75, flush=True)

    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    run()
