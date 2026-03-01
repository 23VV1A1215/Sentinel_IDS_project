import os
import random
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template, request, redirect, url_for, session

# ==========================
# APP INIT
# ==========================

app = Flask(__name__)
app.secret_key = "super_secret_key"

RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")
CURRENT_MODE = "simulation"

# ==========================
# LOAD ML ARTIFACTS
# ==========================

model = joblib.load("models/XGBoost_Tuned.pkl")
encoder = joblib.load("models/encoder.pkl")
scaler = joblib.load("models/scaler.pkl")
numerical_cols = joblib.load("models/numerical_cols.pkl")
categorical_cols = joblib.load("models/categorical_cols.pkl")

# ==========================
# MITRE ATT&CK MAPPING
# ==========================

MITRE_MAPPING = {
    "High": {
        "category": "Denial of Service",
        "mitre_id": "T1499",
        "technique": "Endpoint Denial of Service"
    },
    "Medium": {
        "category": "Network Scanning",
        "mitre_id": "T1046",
        "technique": "Network Service Discovery"
    },
    "Low": {
        "category": "Benign / Low Risk Activity",
        "mitre_id": "N/A",
        "technique": "Normal Traffic"
    }
}

# ==========================
# DATABASE INIT
# ==========================

def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source_ip TEXT,
            total_records INTEGER,
            attacks INTEGER,
            normals INTEGER,
            attack_percentage REAL,
            severity TEXT,
            threat_category TEXT,
            mitre_id TEXT,
            technique TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

# ==========================
# PREPROCESS FUNCTION
# ==========================

def preprocess_input(df):
    df_cat = encoder.transform(df[categorical_cols])
    df_num = scaler.transform(df[numerical_cols])
    return np.hstack((df_num, df_cat))

# ==========================
# DETECTION ENGINE
# ==========================

def run_detection():

    global CURRENT_MODE

    # ---------- Traffic Pattern ----------
    if CURRENT_MODE == "simulation":
        attack_ratio = random.uniform(0.05, 0.75)

    elif CURRENT_MODE == "real":
        if RUNNING_IN_DOCKER:
            return None
        attack_ratio = random.uniform(0.05, 0.25)

    else:  # hybrid
        attack_ratio = random.uniform(0.10, 0.60)

    batch_size = 120
    attack_count = int(batch_size * attack_ratio)

    rows = []

    for i in range(batch_size):

        row = {}
        row["protocol_type"] = random.choice(["tcp", "udp", "icmp"])
        row["service"] = random.choice(["http", "ftp", "smtp", "domain_u"])
        row["flag"] = random.choice(["SF", "REJ", "S0"])

        for col in numerical_cols:
            if i < attack_count:
                row[col] = random.uniform(1.2, 3.0)
            else:
                row[col] = random.uniform(0.0, 0.2)

        rows.append(row)

    df = pd.DataFrame(rows)

    # ---------- ML Inference ----------
    X = preprocess_input(df)
    probs = model.predict_proba(X)[:, 1]

    preds = (probs > 0.75).astype(int)

    total = len(preds)
    attacks = int((preds == 1).sum())
    normals = total - attacks

    attack_percentage = round((attacks / total) * 100, 2)

    # ---------- Severity ----------
    if attack_percentage < 20:
        severity = "Low"
    elif attack_percentage < 50:
        severity = "Medium"
    else:
        severity = "High"

    # ---------- MITRE Mapping ----------
    mitre_info = MITRE_MAPPING.get(severity, MITRE_MAPPING["Low"])

    source_ip = f"192.168.1.{random.randint(1,255)}"

    # ---------- Database Logging ----------
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO logs
        (timestamp, source_ip, total_records, attacks, normals,
         attack_percentage, severity, threat_category, mitre_id, technique)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source_ip,
        total,
        attacks,
        normals,
        attack_percentage,
        severity,
        mitre_info["category"],
        mitre_info["mitre_id"],
        mitre_info["technique"]
    ))

    conn.commit()
    conn.close()

    return {
        "total": total,
        "attacks": attacks,
        "normals": normals,
        "attack_percentage": attack_percentage,
        "severity": severity,
        "source_ip": source_ip,
        "mode": CURRENT_MODE,
        "threat_category": mitre_info["category"],
        "mitre_id": mitre_info["mitre_id"],
        "technique": mitre_info["technique"]
    }

# ==========================
# ROUTES
# ==========================

@app.route('/')
def dashboard():
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "1234":
            session["user"] = "admin"
            return redirect(url_for("dashboard"))
        return "Invalid credentials"
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route('/api/set_mode/<mode>')
def set_mode(mode):
    global CURRENT_MODE

    if mode not in ["simulation", "real", "hybrid"]:
        return jsonify({"error": "Invalid mode"}), 400

    if mode == "real" and RUNNING_IN_DOCKER:
        return jsonify({"error": "Real mode disabled in Docker"}), 403

    CURRENT_MODE = mode
    return jsonify({"status": "Mode switched", "mode": CURRENT_MODE})

@app.route('/api/live')
def live():
    result = run_detection()

    if result is None:
        return jsonify({"error": "Real mode disabled in Docker"}), 403

    return jsonify(result)

@app.route('/api/alert_stats')
def alert_stats():

    conn = sqlite3.connect("predictions.db")

    df = pd.read_sql_query("""
        SELECT severity FROM logs
        WHERE timestamp >= datetime('now','-60 seconds')
    """, conn)

    conn.close()

    return jsonify({
        "high": int((df["severity"] == "High").sum()),
        "medium": int((df["severity"] == "Medium").sum()),
        "low": int((df["severity"] == "Low").sum())
    })

@app.route('/api/logs')
def logs():

    conn = sqlite3.connect("predictions.db")

    df = pd.read_sql_query("""
        SELECT timestamp, source_ip, attack_percentage,
               severity, threat_category, mitre_id, technique
        FROM logs
        ORDER BY id DESC
        LIMIT 10
    """, conn)

    conn.close()

    return jsonify(df.to_dict(orient="records"))

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)