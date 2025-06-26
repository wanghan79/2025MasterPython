import sqlite3
import pickle
from pathlib import Path
from config.settings import Config
from datetime import datetime


class FaceDatabase:
    """人脸数据库管理类"""

    def __init__(self, db_path=None):
        db_path = db_path or Config.DATABASE_URI.replace('sqlite:///', '')
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        """创建数据库表"""
        cursor = self.conn.cursor()

        # 创建人脸表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 创建日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                face_id INTEGER,
                name TEXT,
                confidence REAL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(face_id) REFERENCES faces(id)
            )
        ''')

        # 创建更新时间触发器
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS update_face_timestamp 
            AFTER UPDATE ON faces 
            BEGIN
                UPDATE faces SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
            END;
        ''')
        self.conn.commit()

    def add_face(self, name, encoding, image_path=None):
        """添加新人脸记录"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO faces (name, encoding, image_path) VALUES (?, ?, ?)
        ''', (name, encoding, image_path))
        self.conn.commit()
        return cursor.lastrowid

    def get_face(self, face_id):
        """获取单个人脸记录"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM faces WHERE id = ?', (face_id,))
        return cursor.fetchone()

    def get_all_faces(self):
        """获取所有人脸记录"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM faces ORDER BY name')
        return cursor.fetchall()

    def update_face(self, face_id, name=None, encoding=None, image_path=None):
        """更新人脸记录"""
        cursor = self.conn.cursor()
        updates = []
        params = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if encoding is not None:
            updates.append("encoding = ?")
            params.append(encoding)
        if image_path is not None:
            updates.append("image_path = ?")
            params.append(image_path)

        if not updates:
            return False

        params.append(face_id)
        query = f"UPDATE faces SET {', '.join(updates)} WHERE id = ?"
        cursor.execute(query, params)
        self.conn.commit()
        return cursor.rowcount > 0

    def delete_face(self, face_id):
        """删除人脸记录"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def search_faces(self, name):
        """搜索人脸记录"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM faces WHERE name LIKE ? ORDER BY name
        ''', (f'%{name}%',))
        return cursor.fetchall()

    def add_recognition_log(self, face_id=None, name=None, confidence=None, image_path=None):
        """添加识别日志"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO recognition_logs (face_id, name, confidence, image_path)
            VALUES (?, ?, ?, ?)
        ''', (face_id, name, confidence, image_path))
        self.conn.commit()
        return cursor.lastrowid

    def get_recognition_logs(self, limit=100):
        """获取识别日志"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM recognition_logs 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        return cursor.fetchall()

    def __del__(self):
        """析构时关闭连接"""
        self.conn.close()