"""
MedAd User Authentication Models
================================
SQLite database models for user authentication and chat history persistence.
"""

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    name = db.Column(db.String(100), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to chat history
    chat_sessions = db.relationship('ChatSession', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set the password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if password matches"""
        return check_password_hash(self.password_hash, password)
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def get_chat_history(self, limit=50):
        """Get user's recent chat history"""
        messages = ChatMessage.query.join(ChatSession).filter(
            ChatSession.user_id == self.id
        ).order_by(ChatMessage.timestamp.desc()).limit(limit).all()
        
        # Reverse to get chronological order
        messages.reverse()
        
        # Convert to conversation format used by the app
        conversation = []
        for msg in messages:
            entry = {
                'role': msg.role,
                'content': msg.content,
                'time': msg.timestamp.strftime("%H:%M")
            }
            if msg.role == 'ai':
                entry['data'] = json.loads(msg.medicine_data) if msg.medicine_data else None
                entry['is_emergency'] = msg.is_emergency
                entry['gemini_advice'] = msg.gemini_advice
            conversation.append(entry)
        
        return conversation
    
    def __repr__(self):
        return f'<User {self.email}>'


class ChatSession(db.Model):
    """Chat session for grouping messages"""
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to messages
    messages = db.relationship('ChatMessage', backref='session', lazy='dynamic', cascade='all, delete-orphan')
    
    def add_message(self, role, content, medicine_data=None, is_emergency=False, gemini_advice=None):
        """Add a message to this session"""
        msg = ChatMessage(
            session_id=self.id,
            role=role,
            content=content,
            medicine_data=json.dumps(medicine_data) if medicine_data else None,
            is_emergency=is_emergency,
            gemini_advice=gemini_advice
        )
        db.session.add(msg)
        self.last_activity = datetime.utcnow()
        db.session.commit()
        return msg
    
    def __repr__(self):
        return f'<ChatSession {self.id} for User {self.user_id}>'


class ChatMessage(db.Model):
    """Individual chat message"""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('chat_sessions.id'), nullable=False, index=True)
    role = db.Column(db.String(10), nullable=False)  # 'user' or 'ai'
    content = db.Column(db.Text, nullable=False)
    medicine_data = db.Column(db.Text, nullable=True)  # JSON string of medicine recommendations
    is_emergency = db.Column(db.Boolean, default=False)
    gemini_advice = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<ChatMessage {self.id} ({self.role})>'


def init_db(app):
    """Initialize the database with the Flask app"""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("✅ Database initialized successfully")


def get_or_create_session(user_id):
    """Get the active session or create a new one for a user"""
    # Get the most recent session from today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    session = ChatSession.query.filter(
        ChatSession.user_id == user_id,
        ChatSession.last_activity >= today_start
    ).order_by(ChatSession.last_activity.desc()).first()
    
    if session is None:
        session = ChatSession(user_id=user_id)
        db.session.add(session)
        db.session.commit()
        print(f"📝 Created new chat session for user {user_id}")
    
    return session


def save_message(user_id, role, content, medicine_data=None, is_emergency=False, gemini_advice=None):
    """Save a chat message for a user"""
    session = get_or_create_session(user_id)
    return session.add_message(role, content, medicine_data, is_emergency, gemini_advice)
