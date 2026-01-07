import sqlite3
from datetime import datetime
import json
import os

DATABASE_PATH = "stress_history.db"

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create assessments table (survey-based)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_id TEXT NOT NULL,
            stress_level INTEGER NOT NULL,
            stress_text TEXT NOT NULL,
            inputs JSON NOT NULL,
            top_factors JSON NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create face assessments table (emotion-based)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_id TEXT NOT NULL,
            emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            stress_level INTEGER NOT NULL,
            stress_text TEXT NOT NULL,
            stress_score INTEGER NOT NULL,
            probabilities JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create text assessments table (sentiment-based)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            session_id TEXT NOT NULL,
            text_input TEXT NOT NULL,
            sentiment_polarity REAL NOT NULL,
            sentiment_subjectivity REAL NOT NULL,
            stress_level INTEGER NOT NULL,
            stress_text TEXT NOT NULL,
            stress_score INTEGER NOT NULL,
            keywords_found JSON,
            insights JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_assessment(username, session_id, stress_level, stress_text, inputs, top_factors):
    """Save a stress assessment to the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO assessments (username, session_id, stress_level, stress_text, inputs, top_factors, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        username,
        session_id,
        stress_level,
        stress_text,
        json.dumps(inputs),
        json.dumps(top_factors),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    assessment_id = cursor.lastrowid
    conn.close()
    
    return assessment_id

def get_user_history(session_id, limit=10):
    """Get assessment history for a user session"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, stress_level, stress_text, inputs, top_factors, created_at
        FROM assessments
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (session_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'username': row[1],
            'stress_level': row[2],
            'stress_text': row[3],
            'inputs': json.loads(row[4]),
            'top_factors': json.loads(row[5]),
            'created_at': row[6]
        })
    
    return history

def get_assessment_by_id(assessment_id):
    """Get a specific assessment by ID"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, stress_level, stress_text, inputs, top_factors, created_at
        FROM assessments
        WHERE id = ?
    ''', (assessment_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            'id': row[0],
            'username': row[1],
            'stress_level': row[2],
            'stress_text': row[3],
            'inputs': json.loads(row[4]),
            'top_factors': json.loads(row[5]),
            'created_at': row[6]
        }
    return None

def get_trend_data(session_id, limit=20):
    """Get trend data for charting"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT stress_level, created_at
        FROM assessments
        WHERE session_id = ?
        ORDER BY created_at ASC
        LIMIT ?
    ''', (session_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [{'stress_level': row[0], 'date': row[1]} for row in rows]


# -------------------------
# Face Assessment Functions
# -------------------------

def save_face_assessment(username, session_id, emotion, confidence, stress_level, 
                         stress_text, stress_score, probabilities=None):
    """Save a face/emotion assessment to the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO face_assessments 
        (username, session_id, emotion, confidence, stress_level, stress_text, 
         stress_score, probabilities, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        username,
        session_id,
        emotion,
        confidence,
        stress_level,
        stress_text,
        stress_score,
        json.dumps(probabilities) if probabilities else None,
        datetime.now().isoformat()
    ))
    
    conn.commit()
    assessment_id = cursor.lastrowid
    conn.close()
    
    return assessment_id


def get_face_history(session_id, limit=10):
    """Get face assessment history for a user session"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, emotion, confidence, stress_level, stress_text, 
               stress_score, probabilities, created_at
        FROM face_assessments
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (session_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'username': row[1],
            'emotion': row[2],
            'confidence': row[3],
            'stress_level': row[4],
            'stress_text': row[5],
            'stress_score': row[6],
            'probabilities': json.loads(row[7]) if row[7] else None,
            'created_at': row[8],
            'type': 'face'
        })
    
    return history


# -------------------------
# Text Assessment Functions
# -------------------------

def save_text_assessment(username, session_id, text_input, sentiment_polarity,
                         sentiment_subjectivity, stress_level, stress_text,
                         stress_score, keywords_found=None, insights=None):
    """Save a text/sentiment assessment to the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO text_assessments 
        (username, session_id, text_input, sentiment_polarity, sentiment_subjectivity,
         stress_level, stress_text, stress_score, keywords_found, insights, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        username,
        session_id,
        text_input[:500],  # Truncate long text
        sentiment_polarity,
        sentiment_subjectivity,
        stress_level,
        stress_text,
        stress_score,
        json.dumps(keywords_found) if keywords_found else None,
        json.dumps(insights) if insights else None,
        datetime.now().isoformat()
    ))
    
    conn.commit()
    assessment_id = cursor.lastrowid
    conn.close()
    
    return assessment_id


def get_text_history(session_id, limit=10):
    """Get text assessment history for a user session"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, text_input, sentiment_polarity, sentiment_subjectivity,
               stress_level, stress_text, stress_score, keywords_found, insights, created_at
        FROM text_assessments
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    ''', (session_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'username': row[1],
            'text_input': row[2],
            'sentiment_polarity': row[3],
            'sentiment_subjectivity': row[4],
            'stress_level': row[5],
            'stress_text': row[6],
            'stress_score': row[7],
            'keywords_found': json.loads(row[8]) if row[8] else None,
            'insights': json.loads(row[9]) if row[9] else None,
            'created_at': row[10],
            'type': 'text'
        })
    
    return history


# -------------------------
# Combined History Functions
# -------------------------

def get_all_history(session_id, limit=20):
    """Get all assessment types combined and sorted by date"""
    survey_history = get_user_history(session_id, limit)
    face_history = get_face_history(session_id, limit)
    text_history = get_text_history(session_id, limit)
    
    # Add type to survey assessments
    for item in survey_history:
        item['type'] = 'survey'
    
    # Combine all
    all_history = survey_history + face_history + text_history
    
    # Sort by created_at descending
    all_history.sort(key=lambda x: x['created_at'], reverse=True)
    
    return all_history[:limit]


def get_all_trend_data(session_id, limit=30):
    """Get trend data from all assessment types"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Get survey trends
    cursor.execute('''
        SELECT stress_level, created_at, 'survey' as type
        FROM assessments WHERE session_id = ?
        UNION ALL
        SELECT stress_level, created_at, 'face' as type
        FROM face_assessments WHERE session_id = ?
        UNION ALL
        SELECT stress_level, created_at, 'text' as type
        FROM text_assessments WHERE session_id = ?
        ORDER BY created_at ASC
        LIMIT ?
    ''', (session_id, session_id, session_id, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [{'stress_level': row[0], 'date': row[1], 'type': row[2]} for row in rows]


# Initialize database on module load
init_db()
