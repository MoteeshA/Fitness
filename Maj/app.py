from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB_NAME = "users.db"

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
    # Root will go to dashboard
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
            c.execute("INSERT INTO users (name, email, phone, password) VALUES (?,?,?,?)",
                      (name, email, phone, password))
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

# --- Assessment functionality ---
@app.route("/assessment", methods=["GET", "POST"])
def assessment():
    result = None
    if request.method == "POST":
        age = int(request.form.get("age"))
        gender = request.form.get("gender")
        height = float(request.form.get("height"))
        weight = float(request.form.get("weight"))
        activity = request.form.get("activity")
        goals = request.form.get("goals")
        conditions = request.form.get("conditions")

        # BMI calculation
        bmi = round(weight / ((height / 100) ** 2), 1)

        if bmi < 18.5:
            status = "Underweight"
            score = 60
        elif 18.5 <= bmi < 25:
            status = "Fit"
            score = 85
        elif 25 <= bmi < 30:
            status = "Overweight"
            score = 70
        else:
            status = "Obese"
            score = 50

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

    return render_template("assessment.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
