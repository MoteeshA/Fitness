from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, current_app
import sqlite3
import os
import base64
from io import BytesIO
from PIL import Image
import json
import datetime

# ==== OpenAI ====
from openai import OpenAI, OpenAIError

app = Flask(__name__)
app.secret_key = "supersecretkey"  # move to env in production

DB_NAME = "users.db"

# ---------- OpenAI config (read from env; NO hardcoded keys) ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "").strip() or None
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    # Fail early with a helpful message in the UI / logs
    print("WARNING: OPENAI_API_KEY is not set. Vision features will not work.")

client_kwargs = {"api_key": OPENAI_API_KEY}
if OPENAI_PROJECT:
    client_kwargs["project"] = OPENAI_PROJECT
client = OpenAI(**client_kwargs)


# --- Database Setup ---
def init_db():
    """
    Creates `users` and `logs` tables if DB does not exist.
    logs table stores JSON blobs for assessment/nutrition entries
    linked to users.id (user_id may be NULL for anonymous).
    """
    if not os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        # users table
        c.execute("""
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password TEXT NOT NULL
        )
        """)
        # logs table for assessments and nutrition
        c.execute("""
        CREATE TABLE logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT NOT NULL,
            type TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """)
        conn.commit()
        conn.close()

init_db()


# --- Helpers ---
def get_current_user_id():
    """
    Return DB user id for session["user"] (which stores name in your app).
    Returns None if not logged in or not found.
    """
    if "user" not in session:
        return None
    username = session["user"]
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE name=?", (username,))
    row = c.fetchone()
    conn.close()
    return row[0] if row else None


# --- Routes ---
@app.route("/")
def home():
    return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    """
    Handles GET (show dashboard) and POST (form submission).
    If POST goes here, redirect it to analyze() for processing so url_for('analyze')
    works from dashboard forms.
    """
    if request.method == "POST":
        return redirect(url_for("analyze"))
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
            session["user"] = user[1]  # username stored in second column
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


# --- Assessment Helper ---
def build_result_from_inputs(height, weight, status=None, score=None):
    """Helper to ensure result always has summary and recommendations keys."""
    # defensive: avoid division by zero
    try:
        bmi = round(weight / ((height / 100) ** 2), 1)
    except Exception:
        bmi = None

    if bmi is None:
        status = status or "Unknown"
        score = score or 0
    else:
        if status is None or score is None:
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
        ],
        "Unknown": [
            "Provide valid height and weight for assessment."
        ]
    }

    recommendations = rec_map.get(status, ["Maintain a healthy lifestyle."])
    summary = f"Your BMI is {bmi}. Status: {status}. Recommended: {len(recommendations)} actions." if bmi is not None else "Invalid inputs."

    return {
        "status": status,
        "bmi": bmi,
        "score": score,
        "recommendations": recommendations,
        "summary": summary
    }


# --- Assessment Page (standalone) ---
@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    """
    GET: renders assessment page and passes past logs for calendar display (if logged in).
    POST: processes form (legacy support) — forms may post here or to /run_assessment.
    """
    assessment_obj = None
    logs = []

    # If POST to this route (some templates might post to assessment), handle it
    if request.method == "POST":
        try:
            age = int(request.form.get("age") or 0)
            gender = request.form.get("gender")
            height = float(request.form.get("height") or 0)
            weight = float(request.form.get("weight") or 0)
            activity = request.form.get("activity")
            goals = request.form.get("goals")
            conditions = request.form.get("conditions")

            if height <= 0 or weight <= 0:
                flash("Please provide valid height and weight.", "danger")
            else:
                assessment_obj = build_result_from_inputs(height, weight)
                # Save log
                user_id = get_current_user_id()
                try:
                    conn = sqlite3.connect(DB_NAME)
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO logs (user_id, date, type, data) VALUES (?,?,?,?)",
                        (user_id, datetime.date.today().isoformat(), "assessment", json.dumps(assessment_obj))
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    # don't crash the page if logging fails — just flash a notice
                    print("Warning: failed to save assessment log:", e)
        except Exception as e:
            flash(f"Error in assessment: {e}", "danger")

    # Always pass logs for the calendar if user logged-in
    user_id = get_current_user_id()
    if user_id:
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("SELECT date, type, data FROM logs WHERE user_id=? ORDER BY date DESC", (user_id,))
            rows = c.fetchall()
            conn.close()
            logs = [{"date": r[0], "type": r[1], "data": json.loads(r[2])} for r in rows]
        except Exception as e:
            print("Warning: failed to load logs for assessment:", e)

    return render_template("assessment.html", assessment=assessment_obj, logs=logs)


# --- NEW: route matching the template form action ---
@app.route("/run_assessment", methods=["POST"])
def run_assessment():
    """
    This endpoint exists so templates that call url_for('run_assessment')
    will succeed. It re-uses build_result_from_inputs and returns the
    same 'assessment' variable (keeps parity with your assessment.html).
    Also saves the result into logs (if logged in).
    """
    assessment_obj = None
    try:
        # Accept the same fields your assessment form posts
        age = request.form.get("age")
        gender = request.form.get("gender")
        # convert defensively (allow blank)
        height = float(request.form.get("height") or 0)
        weight = float(request.form.get("weight") or 0)
        activity = request.form.get("activity")
        goals = request.form.get("goals")
        conditions = request.form.get("conditions")

        if height <= 0 or weight <= 0:
            flash("Please provide valid height and weight.", "danger")
            return redirect(url_for("assessment"))

        assessment_obj = build_result_from_inputs(height, weight)

        # Save to logs table for the current user (if any)
        user_id = get_current_user_id()
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute(
                "INSERT INTO logs (user_id, date, type, data) VALUES (?,?,?,?)",
                (user_id, datetime.date.today().isoformat(), "assessment", json.dumps(assessment_obj))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            # don't break if logging fails
            print("Warning: failed to save run_assessment log:", e)

        # Render the assessment page with the assessment object and logs
        # re-query logs so calendar shows latest entry
        logs = []
        uid = get_current_user_id()
        if uid:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("SELECT date, type, data FROM logs WHERE user_id=? ORDER BY date DESC", (uid,))
            rows = c.fetchall()
            conn.close()
            logs = [{"date": r[0], "type": r[1], "data": json.loads(r[2])} for r in rows]

        return render_template("assessment.html", assessment=assessment_obj, logs=logs)

    except Exception as e:
        # keep behavior consistent with the rest of your app
        flash(f"Failed to run assessment: {e}", "danger")
        return redirect(url_for("assessment"))


# --- Analyze route for the dashboard form ---
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Accept defensive conversions; forms might post blank strings
        age = request.form.get("age")
        gender = request.form.get("gender")
        height = float(request.form.get("height") or 0)
        weight = float(request.form.get("weight") or 0)
        activity = request.form.get("activity")
        goals = request.form.get("goals")
        conditions = request.form.get("conditions")

        result = build_result_from_inputs(height, weight)
        # If you want to pass the result to dashboard.html
        return render_template("dashboard.html", result=result)

    except Exception as e:
        flash(f"Error in analyze: {e}", "danger")
        return redirect(url_for("dashboard"))


# --- Nutrition Analysis (upload-based) ---
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
                # try next model in list
                continue
            # other OpenAI errors should be raised
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
        total_protein = sum(float(item.get("protein", 0)) for item in food_items)
        total_carbs = sum(float(item.get("carbs", 0)) for item in food_items)
        total_fat = sum(float(item.get("fat", 0)) for item in food_items)

        # Save nutrition log to DB for current user (if any)
        user_id = get_current_user_id()
        try:
            log_payload = {
                "items": food_items,
                "totals": {
                    "calories": total_calories,
                    "protein": total_protein,
                    "carbs": total_carbs,
                    "fat": total_fat
                }
            }
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute(
                "INSERT INTO logs (user_id, date, type, data) VALUES (?,?,?,?)",
                (user_id, datetime.date.today().isoformat(), "nutrition", json.dumps(log_payload))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print("Warning: failed to save nutrition log:", e)

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
            return jsonify({"ok": True, "is_food": False, "items": [], "totals": {"calories": 0, "protein": 0, "carbs": 0, "fat": 0}})

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


# --- Progress page (reads logs for the current user) ---
@app.route("/progress", methods=["GET"])
def progress():
    logs = []
    series = {"assessments": [], "nutrition": []}

    user_id = get_current_user_id()
    if user_id:
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("SELECT date, type, data FROM logs WHERE user_id=? ORDER BY date ASC", (user_id,))
            rows = c.fetchall()
            conn.close()

            for d, t, data_json in rows:
                parsed = json.loads(data_json)
                logs.append({"date": d, "type": t, "data": parsed})
                if t == "assessment":
                    # include date and bmi/score if present for charting
                    series["assessments"].append({"date": d, "bmi": parsed.get("bmi"), "score": parsed.get("score")})
                elif t == "nutrition":
                    totals = parsed.get("totals") or {}
                    series["nutrition"].append({"date": d, "totals": totals})
        except Exception as e:
            print("Warning: failed to load progress logs:", e)

    return render_template("progress.html", logs=logs, series=series)


# ------- WORKOUTS ROUTE (ADDED to fix your BuildError) -------
@app.route("/workouts", methods=["GET"])
def workouts():
    """
    Minimal workouts page so url_for('workouts') resolves in templates.
    Replace the template content with your actual workouts UI when ready.
    """
    # If you want to show user-specific workouts, fetch them here using get_current_user_id()
    user_id = get_current_user_id()
    sample_workouts = [
        {"title": "Full Body Strength", "duration": "45 min", "level": "Intermediate"},
        {"title": "Morning Yoga Flow", "duration": "30 min", "level": "Beginner"},
        {"title": "HIIT Cardio Blast", "duration": "20 min", "level": "Advanced"},
    ]
    return render_template("workouts.html", workouts=sample_workouts, user_id=user_id)


if __name__ == "__main__":
    app.run(debug=True)
