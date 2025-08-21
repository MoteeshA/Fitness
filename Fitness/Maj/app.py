from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import sqlite3
import os
import base64
from io import BytesIO
from PIL import Image
import json

# ==== OpenAI ====
from openai import OpenAI, OpenAIError

app = Flask(__name__)
app.secret_key = "supersecretkey"  # keep or move to env

DB_NAME = "users.db"

# ---------- OpenAI config (read from env; NO hardcoded keys) ----------
# Set these in your shell or .env:
#   export OPENAI_API_KEY="sk-..."
#   export OPENAI_PROJECT="proj_..."        # optional; only if using project keys
#   export OPENAI_VISION_MODEL="gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "").strip() or None
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    # Fail early with a helpful message in the UI
    print("WARNING: OPENAI_API_KEY is not set. Vision features will not work.")

client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_PROJECT:
    client_kwargs["project"] = OPENAI_PROJECT
client = OpenAI(**client_kwargs)

# --- Database Setup ---
def init_db():
    if not os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password TEXT NOT NULL
        )
        """)
        conn.commit()
        conn.close()

init_db()

# --- Routes ---
@app.route("/")
def home():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["user"] = user[1]  # username
            flash("Login successful!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials!", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        phone = request.form.get("phone")
        password = request.form.get("password")

        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (name, email, phone, password) VALUES (?,?,?,?)",
                (name, email, phone, password)
            )
            conn.commit()
            conn.close()
            flash("Account created successfully! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists!", "danger")
            return redirect(url_for("signup"))

    return render_template("signup.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("dashboard"))

# --- Assessment Page (standalone) ---
@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    result = None
    if request.method == "POST":
        try:
            age = int(request.form.get("age"))
            gender = request.form.get("gender")
            height = float(request.form.get("height"))
            weight = float(request.form.get("weight"))
            activity = request.form.get("activity")
            goals = request.form.get("goals")
            conditions = request.form.get("conditions")

            bmi = round(weight / ((height / 100) ** 2), 1)

            if bmi < 18.5:
                status, score = "Underweight", 60
            elif 18.5 <= bmi < 25:
                status, score = "Fit", 85
            elif 25 <= bmi < 30:
                status, score = "Overweight", 70
            else:
                status, score = "Obese", 50

            rec_map = {
                "Underweight": [
                    "Increase calorie intake with nutrient-rich foods",
                    "Add strength training exercises",
                    "Eat protein-rich snacks",
                    "Ensure 7-8 hours of sleep"
                ],
                "Fit": [
                    "Maintain current workout routine",
                    "Continue balanced diet",
                    "Stay hydrated with 8-10 glasses of water",
                    "Incorporate flexibility training like yoga"
                ],
                "Overweight": [
                    "Incorporate 30 minutes of cardio daily",
                    "Reduce processed sugar and fried foods",
                    "Add high-protein meals to diet",
                    "Walk at least 8,000 steps daily"
                ],
                "Obese": [
                    "Consult a fitness coach for tailored program",
                    "Start with low-impact cardio (walking, swimming)",
                    "Gradually reduce portion sizes",
                    "Increase vegetable intake significantly"
                ]
            }

            recommendations = rec_map.get(status, ["Maintain a healthy lifestyle."])

            result = {
                "status": status,
                "bmi": bmi,
                "score": score,
                "recommendations": recommendations
            }
        except Exception as e:
            flash(f"Error in assessment: {e}", "danger")

    return render_template("assessment.html", result=result)

# --- Analyze route for the dashboard form ---
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        age = int(request.form.get("age"))
        gender = request.form.get("gender")
        height = float(request.form.get("height"))
        weight = float(request.form.get("weight"))
        activity = request.form.get("activity")
        goals = request.form.get("goals")
        conditions = request.form.get("conditions")

        bmi = round(weight / ((height / 100) ** 2), 1)

        if bmi < 18.5:
            status, score = "Underweight", 60
        elif 18.5 <= bmi < 25:
            status, score = "Fit", 85
        elif 25 <= bmi < 30:
            status, score = "Overweight", 70
        else:
            status, score = "Obese", 50

        rec_map = {
            "Underweight": [
                "Increase calorie intake with nutrient-rich foods",
                "Add strength training exercises",
                "Eat protein-rich snacks",
                "Ensure 7-8 hours of sleep"
            ],
            "Fit": [
                "Maintain current workout routine",
                "Continue balanced diet",
                "Stay hydrated with 8-10 glasses of water",
                "Incorporate flexibility training like yoga"
            ],
            "Overweight": [
                "Incorporate 30 minutes of cardio daily",
                "Reduce processed sugar and fried foods",
                "Add high-protein meals to diet",
                "Walk at least 8,000 steps daily"
            ],
            "Obese": [
                "Consult a fitness coach for tailored program",
                "Start with low-impact cardio (walking, swimming)",
                "Gradually reduce portion sizes",
                "Increase vegetable intake significantly"
            ]
        }

        recommendations = rec_map.get(status, ["Maintain a healthy lifestyle."])

        result = {
            "status": status,
            "bmi": bmi,
            "score": score,
            "recommendations": recommendations
        }

        return render_template("dashboard.html", result=result)

    except Exception as e:
        flash(f"Error in analyze: {e}", "danger")
        return redirect(url_for("dashboard"))

# --- Nutrition Analysis (page) ---
@app.route("/nutrition", methods=["GET"])
def nutrition():
    return render_template("nutrition.html")

# ------- OpenAI Vision helper --------
def _vision_call_with_model(model_name, img_b64):
    """
    Helper to query a model; raises OpenAIError to be handled by caller.
    Forces JSON output to reduce parsing errors.
    """
    prompt = (
        "You are a nutrition expert. Look at the food photo and respond ONLY as strict JSON with this schema: "
        "{\"is_food\": true|false, \"items\": [{\"name\": string, \"calories\": integer, \"protein\": number, \"carbs\": number, \"fat\": number}]}"
    )

    return client.chat.completions.create(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a nutrition assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ],
            },
        ],
        max_tokens=500,
        temperature=0.2,
    )

def _try_models_with_image_b64(img_b64):
    """Try configured model, then fallback; return parsed JSON dict."""
    if not OPENAI_API_KEY:
        raise OpenAIError("OpenAI key not set on server. Set OPENAI_API_KEY.")
    models_to_try = [VISION_MODEL, "gpt-4o"] if VISION_MODEL != "gpt-4o" else ["gpt-4o", "gpt-4o-mini"]
    last_err = None
    for m in models_to_try:
        try:
            resp = _vision_call_with_model(m, img_b64)
            raw = resp.choices[0].message.content.strip()
            return json.loads(raw)
        except OpenAIError as e:
            last_err = e
            msg = str(e).lower()
            if ("model" in msg) or ("access" in msg) or ("403" in msg):
                continue
            raise
    if last_err:
        raise last_err
    raise OpenAIError("Unknown OpenAI error")

# --- Upload-based nutrition (existing) ---
@app.route("/analyze_nutrition", methods=["POST"])
def analyze_nutrition():
    if "food_image" not in request.files:
        flash("No file uploaded", "danger")
        return redirect(url_for("nutrition"))

    file = request.files["food_image"]
    if file.filename.strip() == "":
        flash("No file selected", "danger")
        return redirect(url_for("nutrition"))

    try:
        # Convert uploaded image to base64 for OpenAI
        img = Image.open(file.stream).convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        data = _try_models_with_image_b64(img_b64)

        if not data.get("is_food"):
            flash("Upload is not valid (not food).", "danger")
            return redirect(url_for("nutrition"))

        food_items = data.get("items", []) or []
        if not food_items:
            flash("Could not detect food items.", "danger")
            return redirect(url_for("nutrition"))

        total_calories = sum(int(item.get("calories", 0)) for item in food_items)
        total_protein  = sum(float(item.get("protein", 0)) for item in food_items)
        total_carbs    = sum(float(item.get("carbs", 0)) for item in food_items)
        total_fat      = sum(float(item.get("fat", 0)) for item in food_items)

        return render_template(
            "nutrition.html",
            food_items=food_items,
            total_calories=total_calories,
            total_protein=total_protein,
            total_carbs=total_carbs,
            total_fat=total_fat,
            image_data=img_b64
        )

    except OpenAIError as e:
        msg = str(e)
        if "model_not_found" in msg or "does not have access" in msg or "403" in msg:
            flash(
                "OpenAI API error: model access denied. "
                "Either switch to a user-scoped API key (sk-…) or enable the model for your project in the Dashboard → Projects → Models.",
                "danger"
            )
        else:
            flash(f"OpenAI API error: {e}", "danger")
        return redirect(url_for("nutrition"))
    except Exception as e:
        flash(f"Error analyzing image: {e}", "danger")
        return redirect(url_for("nutrition"))

# --- NEW: Live camera frame analysis (JSON in/out) ---
@app.route("/analyze_nutrition_frame", methods=["POST"])
def analyze_nutrition_frame():
    """
    Expects JSON: { "image_b64": "<base64 without data URL>" } or with data URL.
    Returns JSON: { ok: bool, error?: str, is_food: bool, items: [...], totals: {...} }
    """
    try:
        payload = request.get_json(silent=True) or {}
        img_b64 = payload.get("image_b64", "")

        if not img_b64:
            return jsonify({"ok": False, "error": "image_b64 missing"}), 400

        # allow 'data:image/jpeg;base64,...'
        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        data = _try_models_with_image_b64(img_b64)

        if not data.get("is_food"):
            return jsonify({"ok": True, "is_food": False, "items": [], "totals": {"calories":0,"protein":0,"carbs":0,"fat":0}})

        items = data.get("items", []) or []
        totals = {
            "calories": int(sum(int(i.get("calories", 0)) for i in items)),
            "protein": float(sum(float(i.get("protein", 0)) for i in items)),
            "carbs": float(sum(float(i.get("carbs", 0)) for i in items)),
            "fat": float(sum(float(i.get("fat", 0)) for i in items)),
        }
        return jsonify({"ok": True, "is_food": True, "items": items, "totals": totals})

    except OpenAIError as e:
        msg = str(e)
        if "model_not_found" in msg or "does not have access" in msg or "403" in msg:
            return jsonify({
                "ok": False,
                "error": "OpenAI model access denied. Switch to a user-scoped API key (sk-…) or enable the model for your Project."
            }), 403
        return jsonify({"ok": False, "error": f"OpenAI error: {e}"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": f"Server error: {e}"}), 500

if __name__ == "__main__":
    # For local dev; in production use a proper WSGI server
    app.run(debug=True)
