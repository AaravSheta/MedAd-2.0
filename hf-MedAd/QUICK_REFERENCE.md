# 🏥 MedAd - Quick Reference

## ⚡ Removed Unused Technologies

```
❌ REMOVED (not used):
  - sklearn.preprocessing.normalize
  - re (regex module)
  - hashlib
  - plotly.express as px (use plotly.graph_objects instead)
  - urllib.parse

✅ KEPT (actively used):
  - Dash (web framework)
  - Pandas + NumPy (data processing)
  - scikit-learn TF-IDF (ML core)
  - Sentence Transformers (semantic search)
  - Plotly (interactive charts)
  - difflib (text matching)
```

---

## 📝 What User Needs to Provide

### **1. Environment Variables (.env)**
```env
GOOGLE_CLIENT_ID=your_base64_encoded_id
GOOGLE_MAPS_API_KEY=your_api_key
GEMINI_API_KEY=your_base64_encoded_key
```

### **2. CSV Data Files**
- ✅ `all_medicine databased.csv` (248K medicines) - INCLUDED
- ✅ `medicine_dataset.csv` (50K items) - INCLUDED

### **3. Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🎯 Main Project File

**PRIMARY FILE: `web.py`**
- Complete web application
- ~3,370 lines
- All-in-one solution with UI + ML

**Supporting files:**
- `medimatch_app.py` - ML engine (pure Python)
- `medimatch_ml.py` - Alternative ML version

---

## 🚀 Run Application

```bash
python web.py
# Opens at http://localhost:7860
```

---

## 📊 Git Status

✅ **Pushed to HuggingFace Spaces**
- All code synchronized
- Latest commits: clean

⚠️ **GitHub Push Failed**
- Needs conflict resolution
- Remote has newer commits

---

## 📦 Tech Stack

| Component | Tech | Status |
|-----------|------|--------|
| Web UI | Dash + Plotly | ✅ |
| ML Search | TF-IDF + Semantic | ✅ |
| Data | CSV + Pandas | ✅ |
| AI | Google Gemini | ✅ |
| Maps | OpenStreetMap | ✅ |

---

**See SETUP_GUIDE.md for detailed information**
