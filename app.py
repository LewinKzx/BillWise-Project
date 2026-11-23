"""BillWise web app. Run:  python app.py  (your browser opens automatically)"""
import json
import os
import threading
import webbrowser

from flask import Flask, jsonify, render_template, request

import config
import predictor

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", threshold=config.EXEMPTION_THRESHOLD)


@app.post("/api/predict")
def api_predict():
    try:
        return jsonify(predictor.predict(request.get_json(force=True)))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400


@app.get("/api/model-info")
def model_info():
    path = os.path.join(config.REPORT_DIR, "metrics.json")
    if not os.path.exists(path):
        return jsonify({"error": "Run python train.py first"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


HOST, PORT = "127.0.0.1", 5000

if __name__ == "__main__":
    predictor.load_model()  # fail early if not trained
    url = f"http://{HOST}:{PORT}"
    # debug mode starts the app twice (reloader); only open the browser once
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        print(f"\n  BillWise is running at {url}  (opening your browser...)\n")
    app.run(host=HOST, port=PORT, debug=True)
