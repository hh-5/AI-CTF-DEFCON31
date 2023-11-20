
import sqlite3
import numpy as np
import requests

def setup_database():
    conn = sqlite3.connect('noise_all_low.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Images (
        id INTEGER PRIMARY KEY,
        image_data BLOB
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Responses (
        id INTEGER PRIMARY KEY,
        class1 REAL,
        class2 REAL,
        class3 REAL,
        class4 REAL,
        class5 REAL,
        class6 REAL,
        class7 REAL,
        class8 REAL,
        FOREIGN KEY(id) REFERENCES Images(id)
    )
    ''')
    conn.commit()
    return conn, cursor
def setup_database_with_name(name):
    conn = sqlite3.connect(name)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Images (
        id INTEGER PRIMARY KEY,
        image_data BLOB
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Responses (
        id INTEGER PRIMARY KEY,
        class1 REAL,
        class2 REAL,
        class3 REAL,
        class4 REAL,
        class5 REAL,
        class6 REAL,
        class7 REAL,
        class8 REAL,
        FOREIGN KEY(id) REFERENCES Images(id)
    )
    ''')
    conn.commit()
    return conn, cursor

def close_database(conn):
    conn.close()
def insert_image_and_response(conn, image_data, logits):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO Images (image_data) VALUES (?)', (sqlite3.Binary(image_data),))
    image_id = cursor.lastrowid
    cursor.execute('''
    INSERT INTO Responses (id, class1, class2, class3, class4, class5, class6, class7, class8)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (image_id, *logits))
    conn.commit()
    return image_id
def get_response_for_image(conn, image_data):
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM Images WHERE image_data = ?', (image_data,))
    image_id = cursor.fetchone()
    if image_id:
        cursor.execute('SELECT class1, class2, class3, class4, class5, class6, class7, class8 FROM Responses WHERE id = ?', (image_id[0],))
        logits = cursor.fetchone()
        return logits
    return None