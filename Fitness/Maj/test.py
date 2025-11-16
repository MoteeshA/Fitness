import os
import uuid
import base64
import json
import datetime

from io import BytesIO
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    jsonify,
    send_from_directory,
)
import sqlite3
from PIL import Image

# ===== OpenAI =====
from openai import OpenAI, OpenAIError

# ===== Workout Analyzer deps =====
import cv2
import numpy as np
import mediapipe as mp

# ============================
# Flask setup
# ============================
app = Flask(__name__)
app.secret_key = "supersecretkey"  # move to env in production

DB_NAME = "users.db"

# ============================
# Workout folders
# ============================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================
# Mediapipe setup (workouts)
# ============================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

PINK = (255, 0, 255)
NEON_CYAN = (255, 255, 0)

# ---------- OpenAI config ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "").strip() or None
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
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
        c.execute(
            """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT NOT NULL,
            password TEXT NOT NULL
        )
        """
        )
        # logs table for assessments and nutrition
        c.execute(
            """
        CREATE TABLE logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT NOT NULL,
            type TEXT NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
        )
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


# ============================
# Workout helper: joint angle
# ============================
def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) at point b given three points a, b, c.
    Each point is (x, y).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


# --- Routes ---
@app.route("/")
def home():
    return redirect(url_for("dashboard"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    """
    Handles GET (show dashboard) and POST (form submission).
    POST goes to analyze() via redirect so /analyze handles it.
    """
    if request.method == "POST":
        return redirect(url_for("analyze"))
    # result is optionally passed when analyze() renders dashboard.html
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
                (name, email, phone, password),
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
            "Ensure 7-8 hours of sleep",
        ],
        "Fit": [
            "Maintain current workout routine",
            "Continue balanced diet",
            "Stay hydrated with 8-10 glasses of water",
            "Incorporate flexibility training like yoga",
        ],
        "Overweight": [
            "Incorporate 30 minutes of cardio daily",
            "Reduce processed sugar and fried foods",
            "Add high-protein meals to diet",
            "Walk at least 8,000 steps daily",
        ],
        "Obese": [
            "Consult a fitness coach for tailored program",
            "Start with low-impact cardio (walking, swimming)",
            "Gradually reduce portion sizes",
            "Increase vegetable intake significantly",
        ],
        "Unknown": ["Provide valid height and weight for assessment."],
    }

    recommendations = rec_map.get(status, ["Maintain a healthy lifestyle."])
    summary = (
        f"Your BMI is {bmi}. Status: {status}. Recommended: {len(recommendations)} actions."
        if bmi is not None
        else "Invalid inputs."
    )

    return {
        "status": status,
        "bmi": bmi,
        "score": score,
        "recommendations": recommendations,
        "summary": summary,
    }


# --- Assessment Page (standalone) ---
@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    assessment_obj = None
    logs = []

    if request.method == "POST":
        try:
            height = float(request.form.get("height") or 0)
            weight = float(request.form.get("weight") or 0)

            if height <= 0 or weight <= 0:
                flash("Please provide valid height and weight.", "danger")
            else:
                assessment_obj = build_result_from_inputs(height, weight)
                user_id = get_current_user_id()
                try:
                    conn = sqlite3.connect(DB_NAME)
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO logs (user_id, date, type, data) VALUES (?,?,?,?)",
                        (
                            user_id,
                            datetime.date.today().isoformat(),
                            "assessment",
                            json.dumps(assessment_obj),
                        ),
                    )
                    conn.commit()
                    conn.close()
                except Exception as e:
                    print("Warning: failed to save assessment log:", e)
        except Exception as e:
            flash(f"Error in assessment: {e}", "danger")

    user_id = get_current_user_id()
    if user_id:
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute(
                "SELECT date, type, data FROM logs WHERE user_id=? ORDER BY date DESC",
                (user_id,),
            )
            rows = c.fetchall()
            conn.close()
            logs = [{"date": r[0], "type": r[1], "data": json.loads(r[2])} for r in rows]
        except Exception as e:
            print("Warning: failed to load logs for assessment:", e)

    return render_template("assessment.html", assessment=assessment_obj, logs=logs)


# --- NEW: route matching assessment page's JS `url_for('run_assessment')` ---
@app.route("/run_assessment", methods=["POST"])
def run_assessment():
    """
    Called via AJAX from assessment.html.
    Returns JSON with a new log so the calendar and recent list update.
    """
    try:
        if request.is_json:
            payload = request.get_json() or {}
            height = float(payload.get("height") or 0)
            weight = float(payload.get("weight") or 0)
            bmi = payload.get("bmi")
        else:
            # Fallback for form POST
            height = float(request.form.get("height") or 0)
            weight = float(request.form.get("weight") or 0)
            bmi = None

        if height <= 0 or weight <= 0:
            return jsonify({"success": False, "message": "Invalid height or weight."}), 400

        assessment_obj = build_result_from_inputs(height, weight)
        if bmi:
            assessment_obj["bmi"] = bmi

        user_id = get_current_user_id()
        today = datetime.date.today().isoformat()

        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute(
                "INSERT INTO logs (user_id, date, type, data) VALUES (?,?,?,?)",
                (user_id, today, "assessment", json.dumps(assessment_obj)),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print("Warning: failed to save run_assessment log:", e)

        new_log = {"date": today, "type": "assessment", "data": assessment_obj}

        return jsonify({"success": True, "log": new_log})

    except Exception as e:
        return jsonify({"success": False, "message": f"Failed to run assessment: {e}"}), 500


# --- Analyze route for dashboard form ---
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        height = float(request.form.get("height") or 0)
        weight = float(request.form.get("weight") or 0)

        if height <= 0 or weight <= 0:
            flash("Please provide valid height and weight.", "danger")
            return redirect(url_for("dashboard"))

        result = build_result_from_inputs(height, weight)
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
    prompt = (
        "You are a nutrition expert. Look at the food photo and respond ONLY as strict JSON with this schema: "
        '{"is_food": true|false, "items": [{"name": string, "calories": integer, "protein": number, "carbs": number, "fat": number}]}'
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
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            },
        ],
        max_tokens=500,
        temperature=0.2,
    )


def _try_models_with_image_b64(img_b64):
    if not OPENAI_API_KEY:
        raise OpenAIError("OpenAI key not set on server. Set OPENAI_API_KEY.")
    models_to_try = [VISION_MODEL, "gpt-4o"] if VISION_MODEL != "gpt-4o" else [
        "gpt-4o",
        "gpt-4o-mini",
    ]
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


# --- Upload-based nutrition ---
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

        user_id = get_current_user_id()
        try:
            log_payload = {
                "items": food_items,
                "totals": {
                    "calories": total_calories,
                    "protein": total_protein,
                    "carbs": total_carbs,
                    "fat": total_fat,
                },
            }
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute(
                "INSERT INTO logs (user_id, date, type, data) VALUES (?,?,?,?)",
                (
                    user_id,
                    datetime.date.today().isoformat(),
                    "nutrition",
                    json.dumps(log_payload),
                ),
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
            image_data=img_b64,
        )

    except OpenAIError as e:
        msg = str(e)
        if (
            "model_not_found" in msg
            or "does not have access" in msg
            or "403" in msg
        ):
            flash(
                "OpenAI API error: model access denied. "
                "Either switch to a user-scoped API key (sk-…) or enable the model for your project.",
                "danger",
            )
        else:
            flash(f"OpenAI API error: {e}", "danger")
        return redirect(url_for("nutrition"))
    except Exception as e:
        flash(f"Error analyzing image: {e}", "danger")
        return redirect(url_for("nutrition"))


# --- Live camera frame nutrition (JSON) ---
@app.route("/analyze_nutrition_frame", methods=["POST"])
def analyze_nutrition_frame():
    try:
        payload = request.get_json(silent=True) or {}
        img_b64 = payload.get("image_b64", "")

        if not img_b64:
            return jsonify({"ok": False, "error": "image_b64 missing"}), 400

        if "," in img_b64:
            img_b64 = img_b64.split(",", 1)[1]

        data = _try_models_with_image_b64(img_b64)

        if not data.get("is_food"):
            return jsonify(
                {
                    "ok": True,
                    "is_food": False,
                    "items": [],
                    "totals": {
                        "calories": 0,
                        "protein": 0,
                        "carbs": 0,
                        "fat": 0,
                    },
                }
            )

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
        if (
            "model_not_found" in msg
            or "does not have access" in msg
            or "403" in msg
        ):
            return jsonify(
                {
                    "ok": False,
                    "error": "OpenAI model access denied. Enable the model or use a user-scoped key.",
                }
            ), 403
        return jsonify({"ok": False, "error": f"OpenAI error: {e}"}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": f"Server error: {e}"}), 500


# --- Progress page (reads logs) ---
@app.route("/progress", methods=["GET"])
def progress():
    logs = []
    series = {"assessments": [], "nutrition": []}

    user_id = get_current_user_id()
    if user_id:
        try:
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute(
                "SELECT date, type, data FROM logs WHERE user_id=? ORDER BY date ASC",
                (user_id,),
            )
            rows = c.fetchall()
            conn.close()

            for d, t, data_json in rows:
                parsed = json.loads(data_json)
                logs.append({"date": d, "type": t, "data": parsed})
                if t == "assessment":
                    series["assessments"].append(
                        {"date": d, "bmi": parsed.get("bmi"), "score": parsed.get("score")}
                    )
                elif t == "nutrition":
                    totals = parsed.get("totals") or {}
                    series["nutrition"].append({"date": d, "totals": totals})
        except Exception as e:
            print("Warning: failed to load progress logs:", e)

    return render_template("progress.html", logs=logs, series=series)


# ============================
# WORKOUT ROUTES + LOGIC
# ============================

@app.route("/workouts", methods=["GET"])
def workouts():
    """
    Workout analyzer page (pink sticks) in FitSmart theme.
    """
    return render_template(
        "workout.html",
        video_url=None,
        reps=None,
        error=None,
        selected_workout="pushup",
    )


@app.route("/upload_workout", methods=["POST"])
def upload_workout():
    workout_type = request.form.get("workout_type", "pushup")

    if "video" not in request.files:
        return render_template(
            "workout.html",
            video_url=None,
            reps=None,
            error="No file part",
            selected_workout=workout_type,
        )

    file = request.files["video"]
    if file.filename == "":
        return render_template(
            "workout.html",
            video_url=None,
            reps=None,
            error="No selected file",
            selected_workout=workout_type,
        )

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        return render_template(
            "workout.html",
            video_url=None,
            reps=None,
            error="Please upload a video file (mp4/avi/mov/mkv)",
            selected_workout=workout_type,
        )

    input_filename = f"{uuid.uuid4().hex}{ext}"
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    file.save(input_path)

    output_filename = f"{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    reps = process_video(input_path, output_path, workout_type)
    video_url = url_for("outputs", filename=output_filename)

    return render_template(
        "workout.html",
        video_url=video_url,
        reps=reps,
        error=None,
        selected_workout=workout_type,
    )


@app.route("/outputs/<path:filename>")
def outputs(filename):
    """Serve processed workout videos."""
    return send_from_directory(OUTPUT_FOLDER, filename)


def process_video(input_path, output_path, workout_type):
    workout_type = (workout_type or "pushup").lower()

    if workout_type == "jumping_jack":
        return process_jumping_jack_video(input_path, output_path)
    elif workout_type == "pullup":
        return process_pullup_video(input_path, output_path)
    elif workout_type == "squat":
        return process_squat_video(input_path, output_path)
    else:
        return process_pushup_video(input_path, output_path)


# === 1) Jumping Jacks ===
def process_jumping_jack_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Could not open input video")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    counter = 0
    stage = "closed"
    feedback_text = "Stand straight, feet together"
    reached_open = False

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        smooth_landmarks=True,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.addWeighted(frame, 0.4, np.zeros_like(frame), 0.6, 0)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w, _ = image.shape
            current_feedback = feedback_text

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                LEFT_SHOULDER = 11
                RIGHT_SHOULDER = 12
                LEFT_WRIST = 15
                RIGHT_WRIST = 16
                LEFT_ANKLE = 27
                RIGHT_ANKLE = 28

                def get_point(idx):
                    lm = landmarks[idx]
                    return int(lm.x * w), int(lm.y * h)

                l_shoulder = get_point(LEFT_SHOULDER)
                r_shoulder = get_point(RIGHT_SHOULDER)
                l_wrist = get_point(LEFT_WRIST)
                r_wrist = get_point(RIGHT_WRIST)
                l_ankle = get_point(LEFT_ANKLE)
                r_ankle = get_point(RIGHT_ANKLE)

                arms_up = (
                    l_wrist[1] < l_shoulder[1] - 20
                    and r_wrist[1] < r_shoulder[1] - 20
                )
                arms_down = (
                    l_wrist[1] > l_shoulder[1] + 40
                    and r_wrist[1] > r_shoulder[1] + 40
                )

                feet_distance = abs(l_ankle[0] - r_ankle[0]) / float(w)

                FEET_APART_THRESH = 0.20
                FEET_TOGETHER_THRESH = 0.10

                feet_apart = feet_distance > FEET_APART_THRESH
                feet_together = feet_distance < FEET_TOGETHER_THRESH

                open_position = arms_up and feet_apart
                closed_position = arms_down and feet_together

                if open_position and stage == "closed":
                    stage = "open"
                    reached_open = True

                if closed_position and stage == "open" and reached_open:
                    stage = "closed"
                    counter += 1
                    reached_open = False

                feedback_messages = []
                if stage == "closed":
                    if not arms_up:
                        feedback_messages.append("Raise your arms overhead")
                    if not feet_apart:
                        feedback_messages.append("Jump your feet wider")
                else:
                    if not arms_down:
                        feedback_messages.append("Bring your arms back down")
                    if not feet_together:
                        feedback_messages.append("Bring your feet together")

                if not feedback_messages:
                    current_feedback = "Form: GOOD ✅"
                else:
                    current_feedback = " / ".join(feedback_messages)

                feedback_text = current_feedback

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=PINK, thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=PINK, thickness=2, circle_radius=2),
                )

                for point in [
                    l_shoulder,
                    r_shoulder,
                    l_wrist,
                    r_wrist,
                    l_ankle,
                    r_ankle,
                ]:
                    cv2.circle(image, point, 7, PINK, -1)

                cv2.putText(
                    image,
                    f"Feet dist: {feet_distance:.2f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    NEON_CYAN,
                    2,
                )
            else:
                feedback_text = "Make sure your FULL BODY is visible"

            cv2.rectangle(image, (0, 0), (340, 80), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"Jacks: {counter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (180, 105, 255),
                2,
            )
            cv2.putText(
                image,
                f"Stage: {stage.upper()}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )

            cv2.rectangle(image, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.putText(
                image,
                feedback_text,
                (10, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (203, 192, 255),
                2,
            )

            out.write(image)

    cap.release()
    out.release()
    return counter


# === 2) Pull-ups ===
def process_pullup_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Could not open input video")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    counter = 0
    stage = "down"
    feedback_text = "Hang from the bar with straight arms"
    reached_top = False

    ELBOW_TOP_ANGLE = 60
    ELBOW_BOTTOM_ANGLE = 150
    MIN_LIFT_PIXELS = 20

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        smooth_landmarks=True,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.addWeighted(frame, 0.4, np.zeros_like(frame), 0.6, 0)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w, _ = image.shape
            current_feedback = feedback_text

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                LEFT_SHOULDER = 11
                LEFT_ELBOW = 13
                LEFT_WRIST = 15
                LEFT_HIP = 23

                def get_point(idx):
                    lm = landmarks[idx]
                    return int(lm.x * w), int(lm.y * h)

                shoulder = get_point(LEFT_SHOULDER)
                elbow = get_point(LEFT_ELBOW)
                wrist = get_point(LEFT_WRIST)
                hip = get_point(LEFT_HIP)

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                wrist_higher_than_shoulder = wrist[1] < (shoulder[1] - MIN_LIFT_PIXELS)

                if (
                    elbow_angle <= ELBOW_TOP_ANGLE
                    and wrist_higher_than_shoulder
                    and stage == "down"
                ):
                    stage = "up"
                    reached_top = True

                if elbow_angle >= ELBOW_BOTTOM_ANGLE and stage == "up" and reached_top:
                    stage = "down"
                    counter += 1
                    reached_top = False

                feedback_messages = []

                if stage == "down" and elbow_angle < ELBOW_BOTTOM_ANGLE - 10:
                    feedback_messages.append("Straighten your ARMS fully at bottom")

                if stage == "up":
                    if elbow_angle > ELBOW_TOP_ANGLE + 10:
                        feedback_messages.append("Bend elbows MORE at top")
                    if not wrist_higher_than_shoulder:
                        feedback_messages.append("Pull HIGHER – get chin over bar")

                if not feedback_messages:
                    current_feedback = "Form: GOOD ✅"
                else:
                    current_feedback = " / ".join(feedback_messages)

                feedback_text = current_feedback

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=PINK, thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=PINK, thickness=2, circle_radius=2),
                )

                for point in [shoulder, elbow, wrist, hip]:
                    cv2.circle(image, point, 7, PINK, -1)

                cv2.putText(
                    image,
                    f"Elbow: {int(elbow_angle)}°",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    NEON_CYAN,
                    2,
                )
            else:
                feedback_text = "Make sure your UPPER BODY and BAR are visible"

            cv2.rectangle(image, (0, 0), (320, 80), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"Reps: {counter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (180, 105, 255),
                2,
            )
            cv2.putText(
                image,
                f"Stage: {stage.upper()}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )

            cv2.rectangle(image, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.putText(
                image,
                feedback_text,
                (10, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (203, 192, 255),
                2,
            )

            out.write(image)

    cap.release()
    out.release()
    return counter


# === 3) Push-ups ===
def process_pushup_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Could not open input video")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    counter = 0
    stage = "up"
    feedback_text = "Start in plank position"
    reached_bottom = False

    ELBOW_DOWN_ANGLE = 75
    ELBOW_UP_ANGLE = 160
    BODY_STRAIGHT_ANGLE = 170

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        smooth_landmarks=True,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.addWeighted(frame, 0.4, np.zeros_like(frame), 0.6, 0)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w, _ = image.shape
            current_feedback = feedback_text

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                LEFT_SHOULDER = 11
                LEFT_ELBOW = 13
                LEFT_WRIST = 15
                LEFT_HIP = 23
                LEFT_KNEE = 25
                LEFT_ANKLE = 27

                def get_point(idx):
                    lm = landmarks[idx]
                    return int(lm.x * w), int(lm.y * h)

                shoulder = get_point(LEFT_SHOULDER)
                elbow = get_point(LEFT_ELBOW)
                wrist = get_point(LEFT_WRIST)
                hip = get_point(LEFT_HIP)
                knee = get_point(LEFT_KNEE)
                ankle = get_point(LEFT_ANKLE)

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                hip_angle = calculate_angle(shoulder, hip, ankle)

                good_body_line = hip_angle >= BODY_STRAIGHT_ANGLE

                if elbow_angle <= ELBOW_DOWN_ANGLE and good_body_line and stage == "up":
                    stage = "down"
                    reached_bottom = True

                if (
                    elbow_angle >= ELBOW_UP_ANGLE
                    and good_body_line
                    and stage == "down"
                    and reached_bottom
                ):
                    stage = "up"
                    counter += 1
                    reached_bottom = False

                feedback_messages = []

                if not good_body_line:
                    feedback_messages.append("Keep your body STRAIGHT")

                if stage == "down" and elbow_angle > ELBOW_DOWN_ANGLE + 5:
                    feedback_messages.append("Go a bit LOWER")

                if stage == "up" and not reached_bottom and elbow_angle < ELBOW_UP_ANGLE - 10:
                    feedback_messages.append("Use FULL range of motion")

                if not feedback_messages:
                    current_feedback = "Form: GOOD ✅"
                else:
                    current_feedback = " / ".join(feedback_messages)

                feedback_text = current_feedback

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=PINK, thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=PINK, thickness=2, circle_radius=2),
                )

                for point in [shoulder, elbow, wrist, hip, knee, ankle]:
                    cv2.circle(image, point, 7, PINK, -1)

                cv2.putText(
                    image,
                    f"Elbow: {int(elbow_angle)}°",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    NEON_CYAN,
                    2,
                )
                cv2.putText(
                    image,
                    f"Hip: {int(hip_angle)}°",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    NEON_CYAN,
                    2,
                )
            else:
                feedback_text = "Make sure your FULL BODY is visible"

            cv2.rectangle(image, (0, 0), (320, 80), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"Reps: {counter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (180, 105, 255),
                2,
            )
            cv2.putText(
                image,
                f"Stage: {stage.upper()}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )

            cv2.rectangle(image, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.putText(
                image,
                feedback_text,
                (10, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (203, 192, 255),
                2,
            )

            out.write(image)

    cap.release()
    out.release()
    return counter


# === 4) Squats ===
def process_squat_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Could not open input video")
        return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    counter = 0
    stage = "up"
    feedback_text = "Stand tall to start"
    reached_bottom = False

    KNEE_DOWN_ANGLE = 80
    KNEE_UP_ANGLE = 160
    CHEST_FOLD_ANGLE = 140

    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        smooth_landmarks=True,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.addWeighted(frame, 0.4, np.zeros_like(frame), 0.6, 0)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            h, w, _ = image.shape
            current_feedback = feedback_text

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                LEFT_SHOULDER = 11
                LEFT_HIP = 23
                LEFT_KNEE = 25
                LEFT_ANKLE = 27

                def get_point(idx):
                    lm = landmarks[idx]
                    return int(lm.x * w), int(lm.y * h)

                shoulder = get_point(LEFT_SHOULDER)
                hip = get_point(LEFT_HIP)
                knee = get_point(LEFT_KNEE)
                ankle = get_point(LEFT_ANKLE)

                knee_angle = calculate_angle(hip, knee, ankle)
                hip_angle = calculate_angle(shoulder, hip, knee)

                if knee_angle <= KNEE_DOWN_ANGLE and stage == "up":
                    stage = "down"
                    reached_bottom = True

                if knee_angle >= KNEE_UP_ANGLE and stage == "down" and reached_bottom:
                    stage = "up"
                    counter += 1
                    reached_bottom = False

                feedback_messages = []

                if stage == "down" and knee_angle > KNEE_DOWN_ANGLE + 10:
                    feedback_messages.append("Go LOWER for full squat")

                if stage == "up" and not reached_bottom and knee_angle < KNEE_UP_ANGLE - 10:
                    feedback_messages.append("Stand fully between reps")

                if hip_angle < CHEST_FOLD_ANGLE:
                    feedback_messages.append("Keep your chest UP")

                if not feedback_messages:
                    current_feedback = "Form: GOOD ✅"
                else:
                    current_feedback = " / ".join(feedback_messages)

                feedback_text = current_feedback

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=PINK, thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=PINK, thickness=2, circle_radius=2),
                )

                for point in [shoulder, hip, knee, ankle]:
                    cv2.circle(image, point, 7, PINK, -1)

                cv2.putText(
                    image,
                    f"Knee: {int(knee_angle)}°",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    NEON_CYAN,
                    2,
                )
                cv2.putText(
                    image,
                    f"Hip: {int(hip_angle)}°",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    NEON_CYAN,
                    2,
                )
            else:
                feedback_text = "Make sure your FULL BODY is visible"

            cv2.rectangle(image, (0, 0), (320, 80), (0, 0, 0), -1)
            cv2.putText(
                image,
                f"Squats: {counter}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (180, 105, 255),
                2,
            )
            cv2.putText(
                image,
                f"Stage: {stage.upper()}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )

            cv2.rectangle(image, (0, h - 70), (w, h), (0, 0, 0), -1)
            cv2.putText(
                image,
                feedback_text,
                (10, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (203, 192, 255),
                2,
            )

            out.write(image)

    cap.release()
    out.release()
    return counter


if __name__ == "__main__":
    app.run(debug=True)
