"""
CloudScan API Server
Flask backend that accepts Terraform file uploads, runs dynamic RGCN inference,
and returns scan results + LLM remediation.
"""

import os
import sys
import json
import shutil
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS

from dynamic_inference import run_dynamic_inference

app = Flask(__name__)
CORS(app)  # Allow front-end dev server to call the API


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "CloudScan API"})


@app.route("/api/scan", methods=["POST"])
def scan():
    """
    Accept uploaded .tf files, run dynamic inference, return results.

    Expects multipart/form-data with one or more files under the key 'files'.
    Optional query params:
      - threshold (int, 1-3): minimum risk level to flag (default 1)
      - remediation (bool): whether to call LLM (default true)
    """
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded. Use the 'files' field."}), 400

    uploaded_files = request.files.getlist("files")
    tf_files = [f for f in uploaded_files if f.filename.endswith(".tf")]

    if not tf_files:
        return jsonify({"error": "No .tf files found in the upload."}), 400

    # Parse query params
    threshold = request.args.get("threshold", 1, type=int)
    enable_remediation = request.args.get("remediation", "true").lower() != "false"

    # Save uploaded files to a temp directory
    tmp_dir = tempfile.mkdtemp(prefix="cloudscan_")
    try:
        for f in tf_files:
            # Preserve subdirectory structure if the browser sends relative paths
            # (e.g., webkitdirectory uploads send "subfolder/file.tf")
            rel_path = f.filename.replace("\\", "/")
            dest_path = os.path.join(tmp_dir, rel_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            f.save(dest_path)

        # Also save any checkov_report.json if uploaded
        for f in uploaded_files:
            if f.filename.endswith("checkov_report.json"):
                dest = os.path.join(tmp_dir, os.path.basename(f.filename))
                f.save(dest)

        # Count what we received
        saved_count = sum(
            1 for root, _, files in os.walk(tmp_dir)
            for fname in files if fname.endswith(".tf")
        )

        # Run dynamic inference
        result = run_dynamic_inference(
            tf_folder_path=tmp_dir,
            risk_threshold=threshold,
            enable_remediation=enable_remediation,
        )

        # Build response
        response = {
            "success": True,
            "files_received": saved_count,
            "graph_stats": result["graph_stats"],
            "risk_summary": _build_risk_summary(result["all_predictions"]),
            "flagged_resources": _sanitize_flagged(result["flagged_resources"]),
            "remediation": result["remediation"],
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_risk_summary(all_predictions):
    """Aggregate risk distribution across all nodes."""
    summary = {"Safe": 0, "Low": 0, "Medium": 0, "High/Critical": 0}
    for p in all_predictions:
        label = p.get("risk_label", "Safe")
        summary[label] = summary.get(label, 0) + 1
    return summary


def _sanitize_flagged(flagged_resources):
    """
    Clean up flagged resources for JSON serialization.
    Config dicts from hcl2 may contain non-serializable objects.
    """
    cleaned = []
    for res in flagged_resources:
        entry = {
            "node_id": res["node_id"],
            "resource_type": res["resource_type"],
            "predicted_risk": res["predicted_risk"],
            "risk_label": res["risk_label"],
        }
        try:
            # Force JSON serialization to catch any weird types
            json.dumps(res.get("config", {}), default=str)
            entry["config"] = res.get("config", {})
        except:
            entry["config"] = str(res.get("config", {}))
        cleaned.append(entry)
    return cleaned


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting CloudScan API on http://localhost:{port}")
    print(f"Model: {os.path.join(os.path.dirname(__file__), 'rgcn_model.pth')}")
    print(f"Node type map: {os.path.join(os.path.dirname(__file__), 'processed', 'node_type_map.pkl')}")
    app.run(host="0.0.0.0", port=port, debug=True)
