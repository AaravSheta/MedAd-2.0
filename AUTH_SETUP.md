# MedAd - User Authentication & Chat History Setup

## 🎯 Overview
This guide explains the new user authentication and chat history persistence features added to MedAd.

## ✨ New Features

### 1. **User Authentication**
- **Email/Password Login**: Users can create accounts and log in with their email
- **Account Registration**: New users can sign up with name, email, and password
- **Guest Mode**: Users can continue as guests without logging in
- **Password Security**: Passwords are hashed using Werkzeug's `generate_password_hash`

### 2. **Chat History Persistence**
- **Automatic Saving**: All chat queries and AI responses are automatically saved to the database
- **History Retrieval**: When a user logs back in, their previous conversations are loaded
- **Session Management**: Each user has multiple chat sessions (one per day by default)
- **Medicine Data Stored**: Medicine recommendations, Gemini AI advice, and emergency status are all persisted

## 📦 Installation

### Required Packages
The following packages have been added to `requirements.txt`:
```
flask-login>=0.6.0
flask-sqlalchemy>=3.1.0
werkzeug>=3.0.0
```

Install them:
```bash
pip install -r requirements.txt
```

## 🗄️ Database Schema

### Tables Created

#### 1. **users**
Stores user account information
```sql
- id (Primary Key)
- email (Unique, indexed)
- password_hash
- name (optional)
- created_at
- last_login
```

#### 2. **chat_sessions**
Groups messages into sessions per user
```sql
- id (Primary Key)
- user_id (Foreign Key to users)
- created_at
- last_activity
```

#### 3. **chat_messages**
Stores individual messages
```sql
- id (Primary Key)
- session_id (Foreign Key to chat_sessions)
- role ('user' or 'ai')
- content (the message text)
- medicine_data (JSON - medicine recommendations)
- is_emergency (boolean)
- gemini_advice (optional AI advice)
- timestamp
```

## 🚀 How It Works

### Login Flow
```
User opens MedAd
    ↓
Sees login/register page
    ↓
User enters email & password (or signs up)
    ↓
Credentials validated
    ↓
Chat history loaded from database
    ↓
Main app displayed with previous conversations
```

### Chat & Persistence Flow
```
User types symptom query
    ↓
Message saved to database (user role)
    ↓
AI generates recommendations
    ↓
AI response + medicine data saved to database
    ↓
Chat displayed on screen
    ↓
On next login, all history is retrieved
```

## 💾 Database File Location
- **SQLite Database**: `medad_users.db` (created in the MedAd-2.0 directory)
- Can be changed via `DATABASE_URL` environment variable

## 🔒 Security Features

1. **Password Hashing**: Uses `werkzeug.security.generate_password_hash` (PBKDF2)
2. **Email Validation**: Emails are normalized (lowercased, stripped)
3. **Unique Emails**: Prevents duplicate account creation
4. **Minimum Password Length**: 6 characters required
5. **Session Management**: Flask-Login handles session security

## 📝 Environment Variables

Create a `.env` file in the MedAd-2.0 directory:
```
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///medad_users.db
GEMINI_API_KEY=your-gemini-key
```

If `SECRET_KEY` is not set, a random one is generated automatically.

## 🔧 File Structure

### New/Modified Files
```
MedAd-2.0/
├── models.py              # NEW: Database models (User, ChatSession, ChatMessage)
├── web.py                 # MODIFIED: Added Flask-Login, auth callbacks, DB saving
├── requirements.txt       # MODIFIED: Added flask-login, flask-sqlalchemy, werkzeug
└── medad_users.db        # AUTO-CREATED: SQLite database
```

## 📚 Key Functions in models.py

### User Model
```python
user = User.query.filter_by(email='user@example.com').first()
user.check_password('password')           # Verify password
user.update_last_login()                  # Update last login timestamp
user.get_chat_history(limit=50)           # Get previous conversations
```

### Chat Management
```python
# Save a message
save_message(
    user_id=1,
    role='user',              # 'user' or 'ai'
    content='I have headache',
    medicine_data=None,       # Optional: list of medicines
    is_emergency=False,
    gemini_advice=None        # Optional: Gemini AI advice
)

# Get or create session
session = get_or_create_session(user_id=1)
```

## 🎨 UI Changes

### Login Page
- Added email/password input fields
- Added "Sign Up" link to toggle to registration form
- "Continue as Guest" button for anonymous users
- Display name and logout button in the header (when logged in)
- User avatar with initials

### Main App Header
- User badge showing name and first letter avatar
- Logout link to return to login page
- Guest mode displays "👤 Guest"

## 🧪 Testing the Feature

### Test Scenario 1: Login & History Persistence
1. Start the app: `python web.py`
2. Go to `http://localhost:7860`
3. Click "Sign Up" and create a new account
4. Search for a symptom (e.g., "headache")
5. View the medicine recommendations
6. Close the browser completely
7. Reopen `http://localhost:7860`
8. Log in with the same email
9. ✅ **RESULT**: Previous query and recommendations should be visible

### Test Scenario 2: Multiple Queries
1. Log in as any user
2. Ask for 5 different symptoms
3. Refresh the page
4. ✅ **RESULT**: All 5 queries should appear in chat history

### Test Scenario 3: Emergency Mode
1. Log in or use guest mode
2. Search for "heart attack"
3. Should see emergency warning
4. Log out and log back in
5. ✅ **RESULT**: Emergency message should be in history marked as emergency

## 🐛 Troubleshooting

### Issue: Database not created
**Solution**: Run the app. Database is auto-created when Flask-SQLAlchemy initializes.

### Issue: "User already exists" error during signup
**Solution**: The email is already registered. Use a different email or reset the database.

### Issue: Chat history not loading
**Solution**: Check that:
- User ID is correctly stored in session store
- Database file exists and is readable
- User actually has chat messages saved

### Issue: "SECRET_KEY not configured"
**Solution**: Add `SECRET_KEY=your-key` to `.env` file (or it will auto-generate one)

## 📊 Statistics
- **Minimum password length**: 6 characters
- **Session timeout**: None (uses Flask session)
- **Chat history limit**: 50 messages per load (can be adjusted)
- **Database type**: SQLite (easily upgradeable to PostgreSQL)

## 🔜 Future Enhancements
- [ ] Google/Microsoft OAuth integration
- [ ] Password reset via email
- [ ] Share chat history with doctor
- [ ] Medication reminders
- [ ] Upload medical reports
- [ ] Multi-device sync

## ✅ Completed Tasks
- ✅ User authentication with email/password
- ✅ Account registration with validation
- ✅ Password hashing and security
- ✅ Database models created
- ✅ Chat history persistence
- ✅ Automatic message saving
- ✅ History retrieval on login
- ✅ Guest mode support
- ✅ User avatar with initials
- ✅ Logout functionality

---

**Last Updated**: January 2026
**Version**: 3.5+ with Authentication
