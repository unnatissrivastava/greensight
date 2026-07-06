from flask import Flask, render_template, request, redirect, url_for
import sqlite3
import os
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "greensight.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "yield_model.pkl")
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

yield_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def analyze_leaf_image(image_path):
    """
    Real pixel analysis (not a neural net, but genuine computation):
    classifies each pixel as healthy-green, stressed-yellow/brown, or other,
    based on actual RGB values, then reports the real ratios found.
    """
    img = Image.open(image_path).convert("RGB").resize((300, 300))
    arr = np.array(img).astype(int)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    total_pixels = arr.shape[0] * arr.shape[1]

    healthy_mask = (g > r) & (g > b) & (g > 60)
    stressed_mask = (r > 100) & (g > 60) & (b < 100) & (r >= g)
    other_mask = ~(healthy_mask | stressed_mask)

    healthy_pct = float(healthy_mask.sum()) / total_pixels * 100
    stressed_pct = float(stressed_mask.sum()) / total_pixels * 100
    other_pct = float(other_mask.sum()) / total_pixels * 100

    if stressed_pct > 30:
        verdict = "High stress detected"
        confidence = round(min(95, 60 + stressed_pct / 2), 1)
    elif stressed_pct > 12:
        verdict = "Mild stress detected"
        confidence = round(min(90, 55 + stressed_pct), 1)
    else:
        verdict = "Healthy"
        confidence = round(min(97, 70 + healthy_pct / 3), 1)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "healthy_pct": round(healthy_pct, 1),
        "stressed_pct": round(stressed_pct, 1),
        "other_pct": round(other_pct, 1),
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        if not name or not email or not message:
            return render_template("contact.html", error="Please fill in every field.")

        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO submissions (name, email, message) VALUES (?, ?, ?)",
            (name, email, message)
        )
        conn.commit()
        conn.close()

        return redirect(url_for("thanks"))

    return render_template("contact.html")


@app.route("/thanks")
def thanks():
    return render_template("thanks.html")


@app.route("/admin")
def admin():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM submissions ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("admin.html", submissions=rows)


@app.route("/tools/yield-predictor", methods=["GET", "POST"])
def yield_predictor():
    result = None
    if request.method == "POST":
        try:
            rainfall = float(request.form.get("rainfall"))
            temperature = float(request.form.get("temperature"))
            soil_moisture = float(request.form.get("soil_moisture"))
            fertilizer = float(request.form.get("fertilizer"))

            prediction = yield_model.predict(
                [[rainfall, temperature, soil_moisture, fertilizer]]
            )[0]
            result = round(float(prediction), 1)
        except (TypeError, ValueError):
            result = None

    return render_template("yield_predictor.html", result=result)


@app.route("/tools/disease-detector", methods=["GET", "POST"])
def disease_detector():
    analysis = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("leaf_image")
        if file and file.filename:
            save_path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(save_path)
            analysis = analyze_leaf_image(save_path)
            image_url = url_for("static", filename=f"uploads/{file.filename}")

    return render_template("disease_detector.html", analysis=analysis, image_url=image_url)

def generate_ndvi_heatmap(image_path, output_path):
    """
    Real computation, honestly scoped: without an actual near-infrared
    camera we can't compute true NDVI. This computes a GREEN-RED index
    from the visible-light photo as a stand-in, then colours it using
    the same brown->amber->green ramp real NDVI maps use.
    """
    img = Image.open(image_path).convert("RGB").resize((300, 300))
    arr = np.array(img).astype(float)
    r, g = arr[:, :, 0], arr[:, :, 1]

    pseudo_ndvi = (g - r) / (g + r + 1e-5)
    pseudo_ndvi = np.clip(pseudo_ndvi, -1, 1)
    t = (pseudo_ndvi + 1) / 2

    stress = np.array([165, 87, 58], dtype=float)
    harvest = np.array([217, 163, 57], dtype=float)
    healthy = np.array([76, 140, 60], dtype=float)

    heatmap = np.zeros((*t.shape, 3))
    lower = t < 0.5
    tl = np.clip(t / 0.5, 0, 1)
    tu = np.clip((t - 0.5) / 0.5, 0, 1)
    for c in range(3):
        lower_val = stress[c] + (harvest[c] - stress[c]) * tl
        upper_val = harvest[c] + (healthy[c] - harvest[c]) * tu
        heatmap[..., c] = np.where(lower, lower_val, upper_val)

    Image.fromarray(heatmap.astype(np.uint8)).save(output_path)

    return {
        "avg_index": round(float(np.mean(pseudo_ndvi)), 3),
        "healthy_pct": round(float(np.mean(t > 0.6) * 100), 1),
        "stressed_pct": round(float(np.mean(t < 0.35) * 100), 1),
    }


@app.route("/tools/ndvi-mapper", methods=["GET", "POST"])
def ndvi_mapper():
    stats = None
    image_url = None
    heatmap_url = None

    if request.method == "POST":
        file = request.files.get("field_image")
        if file and file.filename:
            save_path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(save_path)

            heatmap_name = f"heatmap_{file.filename}"
            heatmap_path = os.path.join(UPLOAD_DIR, heatmap_name)
            stats = generate_ndvi_heatmap(save_path, heatmap_path)

            image_url = url_for("static", filename=f"uploads/{file.filename}")
            heatmap_url = url_for("static", filename=f"uploads/{heatmap_name}")

    return render_template("ndvi_mapper.html", stats=stats, image_url=image_url, heatmap_url=heatmap_url)


@app.route("/demo")
def demo():
    return render_template("demo.html")


if __name__ == "__main__":
    init_db()
    app.run(debug=True)