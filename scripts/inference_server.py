# -*- coding: utf-8 -*-
"""
@file inference_server.py
@description Persistent HTTP inference server running models and optimization searches to avoid per-request OOM
@module scripts
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['absl_minloglevel'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

import sys
import json
import threading
import traceback
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

# Bootstrap local src package imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aeroml.forward import ForwardV3Predictor
from aeroml.reverse import ReverseV3Designer
import aeroml.features as features

# Global model state
predictor = None
designer = None
is_ready = False
loading_error = None

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to support Numpy arrays and float32 types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)

def load_models():
    global predictor, designer, is_ready, loading_error
    try:
        print("[Inference Server] Loading ForwardV3Predictor...", flush=True)
        predictor = ForwardV3Predictor()
        print("[Inference Server] Loading ReverseV3Designer...", flush=True)
        designer = ReverseV3Designer(forward=predictor)
        is_ready = True
        print("[Inference Server] Models loaded. Server ready.", flush=True)
    except Exception as e:
        loading_error = str(e)
        print(f"[Inference Server] ERROR loading models: {e}", flush=True)
        traceback.print_exc()

class InferenceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence standard HTTP request logs to keep stdout clean
        pass

    def do_GET(self):
        if self.path == "/health":
            if loading_error:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": loading_error}).encode("utf-8"))
            elif is_ready:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ready"}).encode("utf-8"))
            else:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "starting"}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global predictor, designer, is_ready
        if not is_ready:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "starting",
                "message": "Model server is still starting up, please try again shortly."
            }).encode("utf-8"))
            return

        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        try:
            req_data = json.loads(post_data.decode('utf-8'))
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "invalid_json", "message": str(e)}).encode("utf-8"))
            return

        if self.path == "/predict":
            self.handle_predict_api(req_data)
        elif self.path == "/optimize":
            self.handle_optimize_api(req_data)
        else:
            self.send_response(404)
            self.end_headers()

    def handle_predict_api(self, req_data):
        try:
            file_path = req_data.get("file_path")
            re_val = req_data.get("re")
            mach_val = req_data.get("mach")

            if not file_path or re_val is None or mach_val is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "missing_fields",
                    "message": "file_path, re, and mach parameters are required."
                }).encode("utf-8"))
                return

            dat_path = Path(file_path)
            geom = features.geometry_representation(dat_path)
            if geom is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "prediction_failed",
                    "message": f"Could not parse valid coordinates from {file_path}"
                }).encode("utf-8"))
                return

            res = predictor._predict_inputs(geom["profile"], geom["scalar"], re_val, mach_val)

            # Reconstruct surface coordinates for plotting
            coords = features.read_dat_file(dat_path)
            if coords is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "prediction_failed",
                    "message": "Failed to read coordinates"
                }).encode("utf-8"))
                return

            coords = features.normalize_coords(coords)
            upper, lower = features.split_upper_lower(coords)
            upper = features.prepare_surface_for_interp(upper)
            lower = features.prepare_surface_for_interp(lower)
            x_grid = features.cosine_spacing(features.N_STATIONS)
            y_upper = np.interp(x_grid, upper[:, 0], upper[:, 1])
            y_lower = np.interp(x_grid, lower[:, 0], lower[:, 1])
            thickness = y_upper - y_lower
            camber = 0.5 * (y_upper + y_lower)

            payload = {
                "predictions": res["predictions"],
                "uncertainty": res["uncertainty"],
                "geometry": {
                    "fingerprint": geom["fingerprint"],
                    "x": x_grid,
                    "y_upper": y_upper,
                    "y_lower": y_lower,
                    "thickness": thickness,
                    "camber": camber
                },
                "mach_warning": {
                    "extrapolated": features.mach_extrapolation_distance(mach_val) > features.MACH_EXTRAPOLATION_THRESHOLD,
                    "nearest_known_mach": min(features.KNOWN_MACH_VALUES, key=lambda m: abs(m - mach_val)),
                    "distance": features.mach_extrapolation_distance(mach_val),
                }
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload, cls=NumpyEncoder).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "prediction_error",
                "message": str(e),
                "trace": traceback.format_exc()
            }).encode("utf-8"))

    def handle_optimize_api(self, req_data):
        try:
            ldmax = req_data.get("ldmax")
            clmax = req_data.get("clmax")
            cdmin = req_data.get("cdmin")
            re_val = req_data.get("re")
            mach_val = req_data.get("mach")
            n_restarts = req_data.get("restarts", 8)
            opt_maxiter = req_data.get("maxiter", 35)

            if ldmax is None or clmax is None or cdmin is None or re_val is None or mach_val is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "missing_fields",
                    "message": "ldmax, clmax, cdmin, re, and mach parameters are required."
                }).encode("utf-8"))
                return

            target = {"LDMax": ldmax, "ClMax": clmax, "CdMin": cdmin}
            flow = {"Re": re_val, "Mach": mach_val}

            results = designer.run_reverse_search(
                target=target,
                flow=flow,
                n_restarts=n_restarts,
                opt_maxiter=opt_maxiter
            )

            # Extract structural candidate data
            candidates_payload = []
            for c in results["candidates"]:
                candidates_payload.append({
                    "label": c["label"],
                    "objective": float(c["objective"]),
                    "passes_uncertainty": bool(c["passes_uncertainty"]),
                    "predictions": {
                        "LDMax": float(c["LDMax_pred"]),
                        "ClMax": float(c["ClMax_pred"]),
                        "CdMin": float(c["CdMin_pred"])
                    },
                    "uncertainty": {
                        "LDMax_std": float(c["LDMax_std"]),
                        "ClMax_std": float(c["ClMax_std"]),
                        "CdMin_std": float(c["CdMin_std"]),
                        "CdMin_rel_std": float(c["CdMin_rel_std"])
                    },
                    "geometry": {
                        "x": c["geometry"]["x"],
                        "y_upper": c["geometry"]["y_upper"],
                        "y_lower": c["geometry"]["y_lower"],
                        "thickness": c["geometry"]["thickness"],
                        "camber": c["geometry"]["camber"]
                    }
                })

            payload = {
                "feasibility": results["feasibility"],
                "candidates": candidates_payload,
                "mach_warning": {
                    "extrapolated": features.mach_extrapolation_distance(mach_val) > features.MACH_EXTRAPOLATION_THRESHOLD,
                    "nearest_known_mach": min(features.KNOWN_MACH_VALUES, key=lambda m: abs(m - mach_val)),
                    "distance": features.mach_extrapolation_distance(mach_val),
                }
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload, cls=NumpyEncoder).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": "optimization_error",
                "message": str(e),
                "trace": traceback.format_exc()
            }).encode("utf-8"))

def main():
    port = int(os.environ.get("INFERENCE_PORT", 8500))
    host = "127.0.0.1"

    # Start model loading thread
    threading.Thread(target=load_models, daemon=True).start()

    server = HTTPServer((host, port), InferenceHandler)
    print(f"[Inference Server] Running on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Inference Server] Stopping...", flush=True)
        server.server_close()

if __name__ == "__main__":
    main()
