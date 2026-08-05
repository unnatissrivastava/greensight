# 🌿 GreenSight

**AI-powered precision agriculture platform  pitch site + working demo tools.**

GreenSight is a concept for a drone-based crop monitoring startup. This repo contains a full pitch website (Home, Features, About, Contact) plus a set of **working, functional tools**, a trained yield-prediction model, a trained disease classifier, and an NDVI-style field mapper, so the idea can actually be demoed, not just described.

---

## 🧱 Tech Stack

- **Backend**: Python, Flask
- **Database:** SQLite (contact form submissions)
- **ML:** scikit-learn (RandomForest  regression + classification)
- **Image processing:** Pillow, NumPy
- **Frontend:** HTML, CSS, vanilla JavaScript (no framework)

---

## 📁 Project Structure

```text
greensight/
├── app.py                  # Flask app
├── train_yield_model.py    # Train yield model
├── yield_model.pkl
├── disease_model.pkl
├── greensight.db
├── templates/
│   ├── index.html
│   ├── features.html
│   ├── about.html
│   ├── contact.html
│   ├── thanks.html
│   ├── admin.html
│   ├── demo.html
│   ├── yield_predictor.html
│   ├── disease_detector.html
│   └── ndvi_mapper.html
├── static/
│   ├── style.css
│   └── script.js
└── uploads/
```


---

## ▶️ Running Locally

```bash
pip install flask scikit-learn pandas numpy pillow joblib
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

To retrain the yield model from scratch:
```bash
python train_yield_model.py
```

---

## 🔧 Features / Routes

| Route | What it does |
|---|---|
| `/` | Home — pitch hero, problem stats |
| `/features` | Six-step growth timeline explaining the product |
| `/about` | Mission, data sourcing, business model |
| `/contact` | Contact form → saved to SQLite → redirects to `/thanks` |
| `/admin` | View all contact form submissions |
| `/demo` | **Live demo dashboard** — links to all three tools below + simulated live sensor readings |
| `/tools/yield-predictor` | Enter rainfall, temperature, soil moisture, fertilizer → get a real model prediction |
| `/tools/disease-detector` | Upload a leaf photo → real trained classifier returns healthy/diseased + confidence |
| `/tools/ndvi-mapper` | Upload a field photo → generates a real per-pixel vegetation heatmap |

---

## 🧠 Honest Note on What's "Real" vs Approximated

This project is upfront about scope — some tools are genuinely trained ML, others are honest approximations given we don't have real drone/IoT hardware:

- **Yield Predictor** — a real `RandomForestRegressor` (scikit-learn), trained on a **synthetic dataset** we generated ourselves (rainfall, temperature, soil moisture, fertilizer → yield, built from a realistic agronomic formula + noise, since a real farm-sensor dataset wasn't available). Test R² ≈ 0.83. Swap in a real dataset by editing `train_yield_model.py`.

- **Disease Detector** — a real `RandomForestClassifier`, trained on real, labeled images from the public **PlantVillage** academic dataset (tomato: healthy vs. Late Blight). Test accuracy ≈ 96%. This is genuinely trained on real photos, not a heuristic.

- **NDVI Mapper** — real computation, but an **approximation**: true NDVI needs a near-infrared camera, which we don't have. Instead this computes a green-vs-red vegetation index from ordinary RGB photos and colours it using the same brown→amber→green scale real NDVI maps use. Clearly labeled as an approximation in the UI.

- **Live sensor readings** on `/demo` are **simulated** (a small random walk in JavaScript) — there's no real IoT hardware connected. This is disclosed in the page footer.

---

## 🗺️ Possible Next Steps

- Swap the synthetic yield dataset for a real agricultural dataset
- Add more disease classes beyond tomato/Late Blight
- Password-protect `/admin`
- Deploy (Render, PythonAnywhere, Railway) for a public link instead of localhost

---

Built as a hackathon / pitch prototype. Not production software.
