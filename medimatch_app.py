"""
MEDIMATCH ML ENGINE - MedAd Machine Learning Core
Hybrid Search: TF-IDF + Semantic (Sentence Transformers)
+ Google Translate (Hindi to English via Gemini AI)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import zipfile
import plotly.graph_objects as go
import urllib.request
import json
from dotenv import load_dotenv

load_dotenv()

# Semantic Search with Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
    print("✅ Sentence Transformers loaded")
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠️ Sentence Transformers not available")

# =============================================================================
# GOOGLE TRANSLATE (Using Gemini AI - FREE, no separate API key needed!)
# =============================================================================
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
TRANSLATE_ENABLED = True

def translate_to_english(text):
    """Translate Hindi/Regional text to English using Gemini AI (FREE)"""
    if not TRANSLATE_ENABLED:
        return text
    
    # Quick check - if already English, return as-is
    if all(ord(c) < 128 or c in ' .,!?' for c in text):
        return text
    
    try:
        prompt = f"""Translate the following text to English. If it's already in English, return it as-is.
Only return the translation, nothing else. Keep medical terms accurate.

Text: {text}

Translation:"""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        
        data = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 100}
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode('utf-8'))
            translated = result['candidates'][0]['content']['parts'][0]['text'].strip()
            print(f"🌐 Translated: '{text}' → '{translated}'")
            return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# DATA LOADING
def extract_zip_if_needed(zip_path, extract_to='.'):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    return False

def load_data():
    """Load medicine datasets (248K + 50K medicines)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    file1 = os.path.join(script_dir, 'all_medicine databased.csv')
    if not os.path.exists(file1):
        print(f"ERROR: 'all_medicine databased.csv' not found")
        return None, None
    
    df1 = pd.read_csv(file1, low_memory=False)

    file2 = os.path.join(script_dir, 'medicine_dataset.csv')
    zip2 = os.path.join(script_dir, 'medicine_dataset.csv.zip')
    
    if not os.path.exists(file2) and os.path.exists(zip2):
        extract_zip_if_needed(zip2, script_dir)
    
    if not os.path.exists(file2):
        print(f"ERROR: 'medicine_dataset.csv' not found")
        return None, None
        
    df2 = pd.read_csv(file2, low_memory=False)
    df1, df2 = preprocess_data(df1, df2)
    
    return df1, df2

def preprocess_data(df1, df2):
    """Preprocess: combine uses, add therapeutic class, clean names"""
    use_cols = [c for c in df1.columns if 'use' in c.lower()]
    df1['combined_use'] = df1[use_cols].fillna('').agg(' '.join, axis=1).str.lower()
    
    if 'Therapeutic Class' in df1.columns:
        df1['combined_use'] = df1['combined_use'] + ' ' + df1['Therapeutic Class'].fillna('').str.lower()
    
    if 'name' in df1.columns:
        df1['name_clean'] = df1['name'].str.lower().str.strip()
    if 'Name' in df2.columns:
        df2['Name_clean'] = df2['Name'].str.lower().str.strip()
    
    return df1, df2

# HELPER FUNCTIONS (used by search algorithms)
def get_medicine_uses(medicine):
    uses = []
    for i in range(10):
        use_col = f'use{i}'
        if use_col in medicine and pd.notna(medicine[use_col]) and medicine[use_col]:
            uses.append(str(medicine[use_col]))
    return ', '.join(uses[:5]) if uses else 'General medicine'

def get_side_effects(medicine):
    effects = []
    for i in range(5):
        effect_col = f'sideEffect{i}'
        if effect_col in medicine and pd.notna(medicine[effect_col]) and medicine[effect_col]:
            effects.append(str(medicine[effect_col]))
    return ', '.join(effects[:3]) if effects else 'Consult doctor'

# TF-IDF MODEL
def train_model(df1):
    """Train TF-IDF: 15K features, n-grams 1-4, sublinear TF, L2 norm"""
    if df1 is None or df1.empty:
        return None, None
    
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=15000,
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.90,
        sublinear_tf=True,
        smooth_idf=True,
        norm='l2'
    )
    
    tfidf_matrix = vectorizer.fit_transform(df1['combined_use'].astype(str))
    print(f"✅ TF-IDF Trained: {len(vectorizer.vocabulary_)} vocab, {tfidf_matrix.shape} matrix")
    
    return vectorizer, tfidf_matrix

# SEMANTIC SEARCH - Sentence Transformers
semantic_model = None
medicine_embeddings = None
medicine_indices = None

def load_semantic_model():
    """Load lightweight sentence transformer: all-MiniLM-L6-v2 (22M params, 80MB)"""
    global semantic_model
    if not SEMANTIC_AVAILABLE:
        print("⚠️ Semantic search not available")
        return None
    
    try:
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Semantic Model Loaded: all-MiniLM-L6-v2")
        return semantic_model
    except Exception as e:
        print(f"❌ Semantic model failed: {e}")
        return None

def create_medicine_embeddings(df1, sample_size=8000):
    """Create embeddings for medicine uses (sampled for speed)"""
    global medicine_embeddings, medicine_indices
    
    if semantic_model is None or df1 is None:
        return None, None
    
    try:
        if len(df1) > sample_size:
            sampled_df = df1.sample(n=sample_size, random_state=42)
        else:
            sampled_df = df1
        
        medicine_indices = sampled_df.index.tolist()
        texts = sampled_df['combined_use'].fillna('').astype(str).tolist()
        
        print(f"🔄 Creating embeddings for {len(texts)} medicines...")
        medicine_embeddings = semantic_model.encode(texts, show_progress_bar=True, batch_size=64)
        print(f"✅ Embeddings created: {medicine_embeddings.shape}")
        
        return medicine_embeddings, medicine_indices
    except Exception as e:
        print(f"❌ Embedding creation failed: {e}")
        return None, None

def semantic_search(query, df1, top_n=50):
    """Semantic similarity search using sentence embeddings"""
    global medicine_embeddings, medicine_indices
    
    if semantic_model is None or medicine_embeddings is None:
        return []
    
    try:
        expanded_query = expand_symptoms(query)
        query_embedding = semantic_model.encode([expanded_query])[0]
        
        similarities = cosine_similarity([query_embedding], medicine_embeddings).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Semantic threshold
                actual_idx = medicine_indices[idx]
                medicine = df1.iloc[actual_idx]
                
                results.append({
                    'index': actual_idx,
                    'Medicine Name': medicine.get('name', medicine.get('Medicine Name', 'Unknown')),
                    'Therapeutic Class': medicine.get('Therapeutic Class', 'General'),
                    'Uses': get_medicine_uses(medicine),
                    'Side Effects': get_side_effects(medicine),
                    'Manufacturer': medicine.get('Manufacturer', 'N/A'),
                    'Semantic Score': similarities[idx]
                })
        
        return results
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []

def hybrid_search(query, df1, vectorizer, tfidf_matrix, top_n=15):
    """Hybrid TF-IDF + Semantic search with score boosting"""
    tfidf_results = search_medicines(query, df1, vectorizer, tfidf_matrix, top_n=100)
    semantic_results = semantic_search(query, df1, top_n=50)
    
    if not semantic_results:
        return tfidf_results[:top_n]
    
    semantic_names = {r['Medicine Name'].lower() for r in semantic_results}
    
    boosted_results = []
    for result in tfidf_results:
        if result['Medicine Name'].lower() in semantic_names:
            result['Raw Score'] = result['Raw Score'] * 1.5
            result['Match Score'] = f"{min(result['Raw Score'] * 100 * 1.5, 99.9):.1f}%"
            result['Search Type'] = 'Hybrid'
        else:
            result['Search Type'] = 'TF-IDF'
        boosted_results.append(result)
    
    for sem_result in semantic_results:
        if not any(r['Medicine Name'].lower() == sem_result['Medicine Name'].lower() for r in boosted_results):
    boosted_results.sort(key=lambda x: x['Raw Score'], reverse=True)
    return boosted_results[:top_n]

# 30 SYMPTOM CATEGORIES WITH MEDICAL SYNONYMS
SYMPTOM_SYNONYMS = {
    'headache': 'headache head pain migraine cephalalgia tension headache cluster headache sinus headache',
    'fever': 'fever pyrexia high temperature febrile hyperthermia chills shivering',
    'cold': 'cold common cold flu influenza runny nose nasal congestion sneezing rhinitis',
    'cough': 'cough dry cough wet cough productive cough bronchitis whooping tussis',
    'pain': 'pain ache soreness discomfort body pain muscle pain joint pain arthralgia myalgia',
    'nausea': 'nausea vomiting upset stomach queasiness motion sickness antiemetic',
    'diarrhea': 'diarrhea loose motion stomach upset gastroenteritis dysentery',
    'allergy': 'allergy allergic reaction itching hives urticaria antihistamine rhinitis',
    'diabetes': 'diabetes blood sugar hyperglycemia glucose insulin antidiabetic',
    'hypertension': 'hypertension high blood pressure bp antihypertensive cardiovascular',
    'infection': 'infection bacterial viral fungal sepsis antibiotic antimicrobial',
    'anxiety': 'anxiety stress tension nervousness panic anxiolytic restless',
    'insomnia': 'insomnia sleeplessness sleep disorder trouble sleeping sedative hypnotic',
    'acidity': 'acidity heartburn acid reflux gastritis gerd antacid dyspepsia',
    'asthma': 'asthma wheezing breathing difficulty bronchospasm inhaler bronchodilator',
    'depression': 'depression sad mood antidepressant serotonin melancholy',
    'skin': 'skin rash eczema dermatitis psoriasis fungal cream ointment topical',
    'eye': 'eye vision conjunctivitis dry eye drops ophthalmic ocular',
    'ear': 'ear pain otitis infection drops otic hearing tinnitus',
    'throat': 'throat sore pharyngitis tonsillitis strep laryngitis',
    'vitamin': 'vitamin supplement deficiency multivitamin nutrition minerals',
    'blood': 'blood anemia iron hemoglobin platelet hematologic',
    'heart': 'heart cardiac arrhythmia angina cardiovascular palpitation',
    'kidney': 'kidney renal urinary nephro bladder',
    'liver': 'liver hepatic hepatitis cirrhosis hepato biliary',
    'thyroid': 'thyroid hypothyroid hyperthyroid levothyroxine goiter',
    'cholesterol': 'cholesterol lipid statin triglyceride hdl ldl',
    'constipation': 'constipation laxative bowel movement stool softener',
    'dizziness': 'dizziness vertigo lightheaded balance spinning faint',
    'fatigue': 'fatigue tiredness weakness energy exhaustion lethargy',
}

# SMART BODY PART + SYMPTOM DETECTION
SMART_SYMPTOM_MAP = {
    ('head', 'pain'): 'headache migraine cephalalgia head pain tension headache',
    ('head', 'paining'): 'headache migraine cephalalgia head pain tension headache',
    ('head', 'ache'): 'headache migraine cephalalgia head pain tension headache',
    ('head', 'hurt'): 'headache migraine cephalalgia head pain tension headache',
    ('sir', 'dard'): 'headache migraine cephalalgia head pain tension headache',
    ('sar', 'dard'): 'headache migraine cephalalgia head pain tension headache',
    ('stomach', 'pain'): 'stomach pain gastric abdominal pain gastritis peptic ulcer acidity',
    ('stomach', 'ache'): 'stomach pain gastric abdominal pain gastritis peptic ulcer acidity',
    ('pet', 'dard'): 'stomach pain gastric abdominal pain gastritis peptic ulcer acidity',
    ('tummy', 'ache'): 'stomach pain gastric abdominal pain gastritis peptic ulcer acidity',
    ('back', 'pain'): 'back pain lumbar backache spinal muscular relaxant',
    ('kamar', 'dard'): 'back pain lumbar backache spinal muscular relaxant',
    ('chest', 'pain'): 'chest pain angina cardiac heart antacid',
    ('throat', 'pain'): 'sore throat pharyngitis tonsillitis strep throat infection',
    ('gala', 'dard'): 'sore throat pharyngitis tonsillitis strep throat infection',
    ('knee', 'pain'): 'knee pain joint pain arthritis orthopedic glucosamine',
    ('ghutna', 'dard'): 'knee pain joint pain arthritis orthopedic glucosamine',
    ('joint', 'pain'): 'joint pain arthritis rheumatoid orthopedic glucosamine',
    ('tooth', 'pain'): 'toothache dental pain analgesic antibiotic dental',
    ('dant', 'dard'): 'toothache dental pain analgesic antibiotic dental',
    ('eye', 'pain'): 'eye pain conjunctivitis ophthalmic eye drops',
    ('ear', 'pain'): 'ear pain otitis otic ear drops infection',
    ('muscle', 'pain'): 'muscle pain myalgia muscular relaxant sprain strain',
    ('body', 'pain'): 'body pain analgesic painkiller fever viral',
}

def smart_symptom_detection(user_input):
    """Detect body part + symptom combinations for precise matching"""
    query = user_input.lower()
    detected_symptoms = []
    
    for (body_part, symptom), expansion in SMART_SYMPTOM_MAP.items():
        if body_part in query and symptom in query:
            detected_symptoms.append(expansion)
    
    if detected_symptoms:
        return ' '.join(detected_symptoms)
    return None

def expand_symptoms(user_input):
    """Expand user input with medical synonyms for better matching"""
    expanded = user_input.lower()
    
    # First try smart detection
    smart_expansion = smart_symptom_detection(user_input)
    if smart_expansion:
        expanded = expanded + ' ' + smart_expansion
        return expanded
    
    # Fallback to keyword-based expansion
    for symptom, synonyms in SYMPTOM_SYNONYMS.items():
        if symptom in expanded:
            expanded = expanded + ' ' + synonyms
    return expanded

# SMART QUERY VALIDATION - Reject non-medical queries
MEDICAL_KEYWORDS = {
    'pain', 'paining', 'ache', 'aching', 'hurt', 'hurting', 'dard',
    'fever', 'cold', 'cough', 'headache', 'stomach', 'nausea', 'vomiting',
    'diarrhea', 'allergy', 'rash', 'infection', 'diabetes', 'sugar', 'blood', 'pressure',
    'heart', 'chest', 'breathing', 'asthma', 'anxiety', 'stress', 'depression', 'sleep',
    'insomnia', 'tired', 'fatigue', 'weakness', 'dizziness', 'vertigo', 'eye', 'ear',
    'throat', 'skin', 'joint', 'muscle', 'back', 'knee', 'leg', 'arm', 'neck', 'shoulder',
    'vitamin', 'supplement', 'tablet', 'medicine', 'drug', 'capsule', 'syrup', 'injection',
    'antibiotic', 'painkiller', 'treatment', 'cure', 'remedy', 'health', 'medical', 'doctor',
    'hospital', 'disease', 'illness', 'symptom', 'sickness', 'flu', 'viral', 'bacterial',
    'fungal', 'wound', 'injury', 'burn', 'cut', 'swelling', 'inflammation', 'cramp',
    'migraine', 'acidity', 'gas', 'bloating', 'constipation', 'digestion', 'liver', 'kidney',
    'thyroid', 'cholesterol', 'weight', 'obesity', 'pregnancy', 'periods', 'menstrual',
    'bone', 'fracture', 'sprain', 'arthritis', 'cancer', 'tumor', 'ulcer', 'hernia',
    'piles', 'hemorrhoids', 'urinary', 'prostate', 'sexual', 'hormonal', 'immunity',
    'covid', 'corona', 'malaria', 'dengue', 'typhoid', 'jaundice', 'hepatitis', 'tb',
    'tuberculosis', 'hiv', 'aids', 'epilepsy', 'seizure', 'paralysis', 'stroke', 'bp',
    'insulin', 'glucose', 'hemoglobin', 'platelet', 'wbc', 'rbc', 'uric',
    'paracetamol', 'ibuprofen', 'aspirin', 'crocin', 'dolo', 'combiflam', 'calpol',
    'azithromycin', 'amoxicillin', 'cetirizine', 'montair', 'pantoprazole', 'omeprazole',
    'metformin', 'amlodipine', 'atorvastatin', 'losartan', 'telmisartan',
    'zinc', 'iron', 'calcium', 'b12', 'd3', 'folic', 'biotin', 'omega', 'protein',
    'dard', 'bukhar', 'khansi', 'zukam', 'sir', 'pet', 'kamar', 'ghutna', 'gala',
    'aankh', 'kaan', 'dant', 'tooth', 'dental', 'oral', 'mouth', 'gum', 'tongue'
}

NON_MEDICAL_PATTERNS = [
    'hi', 'hello', 'hey', 'bye', 'thanks', 'thank', 'ok', 'okay', 'yes', 'no', 'please',
    'help', 'what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'could', 'would',
    'padhai', 'study', 'exam', 'school', 'college', 'job', 'work', 'money', 'salary',
    'weather', 'time', 'date', 'day', 'movie', 'song', 'music', 'game', 'play', 'food',
    'recipe', 'cook', 'travel', 'hotel', 'flight', 'train', 'bus', 'car', 'bike',
    'phone', 'laptop', 'computer', 'internet', 'wifi', 'app', 'website', 'download',
    'love', 'relationship', 'friend', 'family', 'marriage', 'wedding', 'party',
    'news', 'politics', 'sports', 'cricket', 'football', 'ipl', 'match',
    'good', 'bad', 'nice', 'great', 'awesome', 'cool', 'hot', 'beautiful',
    'kaise', 'kya', 'kab', 'kahan', 'kaun', 'kyun', 'accha', 'theek', 'sahi',
    'padhai nhi', 'bore', 'boring', 'lonely', 'sad', 'happy', 'angry', 'hungry',
    'thirsty', 'sleepy', 'lazy', 'busy', 'free', 'available'
]

def is_medical_query(user_input):
    """Check if query is medical-related"""
    query = user_input.lower().strip()
    
    if len(query) < 3:
        return False, "Please describe your health concern in more detail."
    
    query_words = query.split()
    if len(query_words) <= 2:
        for pattern in NON_MEDICAL_PATTERNS:
            if pattern in query:
                return False, "I'm MedAd, a medicine recommendation assistant. Please describe your health symptoms or medical concerns."
    
    has_medical_keyword = False
    for keyword in MEDICAL_KEYWORDS:
        if keyword in query:
            has_medical_keyword = True
            break
    
    if not has_medical_keyword and len(query_words) <= 3:
        return False, "I can only help with health and medicine related queries. Please describe symptoms like 'headache', 'fever', 'stomach pain', etc."
    
    return True, None

# CORE SEARCH ALGORITHM
def search_medicines(query, df1, vectorizer, tfidf_matrix, top_n=15):
    """TF-IDF + Cosine Similarity search with synonym expansion"""
    if vectorizer is None or tfidf_matrix is None or df1 is None:
        return []
    
    expanded_query = expand_symptoms(query)
    query_vector = vectorizer.transform([expanded_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.01:
            medicine = df1.iloc[idx]
            match_score = min(similarities[idx] * 100 * 1.5, 99.9)
            
            results.append({
                'Medicine Name': medicine.get('name', medicine.get('Medicine Name', 'Unknown')),
                'Therapeutic Class': medicine.get('Therapeutic Class', 'General'),
                'Action Class': medicine.get('Action Class', 'N/A'),
                'Uses': get_medicine_uses(medicine),
                'Side Effects': get_side_effects(medicine),
                'Manufacturer': medicine.get('Manufacturer', 'N/A'),
                'Match Score': f"{match_score:.1f}%",
                'Raw Score': similarities[idx]
            })
    
    return results

# DISEASE ANALYTICS
DISEASE_STATS = {
    'headache': {'prevalence': 46, 'recovery_rate': 95, 'avg_duration': 2, 'severity': 'Low'},
    'fever': {'prevalence': 35, 'recovery_rate': 98, 'avg_duration': 3, 'severity': 'Moderate'},
    'cold': {'prevalence': 62, 'recovery_rate': 99, 'avg_duration': 7, 'severity': 'Low'},
    'cough': {'prevalence': 45, 'recovery_rate': 96, 'avg_duration': 10, 'severity': 'Low'},
    'pain': {'prevalence': 50, 'recovery_rate': 90, 'avg_duration': 5, 'severity': 'Moderate'},
    'diabetes': {'prevalence': 10, 'recovery_rate': 70, 'avg_duration': 365, 'severity': 'High'},
    'hypertension': {'prevalence': 25, 'recovery_rate': 75, 'avg_duration': 365, 'severity': 'High'},
    'allergy': {'prevalence': 30, 'recovery_rate': 85, 'avg_duration': 14, 'severity': 'Low'},
    'infection': {'prevalence': 40, 'recovery_rate': 92, 'avg_duration': 7, 'severity': 'Moderate'},
    'anxiety': {'prevalence': 18, 'recovery_rate': 80, 'avg_duration': 90, 'severity': 'Moderate'},
    'asthma': {'prevalence': 8, 'recovery_rate': 85, 'avg_duration': 365, 'severity': 'High'},
    'acidity': {'prevalence': 35, 'recovery_rate': 95, 'avg_duration': 7, 'severity': 'Low'},
    'insomnia': {'prevalence': 20, 'recovery_rate': 82, 'avg_duration': 30, 'severity': 'Moderate'},
    'diarrhea': {'prevalence': 28, 'recovery_rate': 97, 'avg_duration': 3, 'severity': 'Low'},
    'nausea': {'prevalence': 32, 'recovery_rate': 98, 'avg_duration': 2, 'severity': 'Low'},
    'skin': {'prevalence': 22, 'recovery_rate': 88, 'avg_duration': 14, 'severity': 'Low'},
    'eye': {'prevalence': 15, 'recovery_rate': 92, 'avg_duration': 7, 'severity': 'Low'},
    'ear': {'prevalence': 12, 'recovery_rate': 94, 'avg_duration': 7, 'severity': 'Low'},
    'throat': {'prevalence': 38, 'recovery_rate': 96, 'avg_duration': 5, 'severity': 'Low'},
    'heart': {'prevalence': 8, 'recovery_rate': 78, 'avg_duration': 365, 'severity': 'High'},
    'kidney': {'prevalence': 5, 'recovery_rate': 75, 'avg_duration': 180, 'severity': 'High'},
    'liver': {'prevalence': 4, 'recovery_rate': 72, 'avg_duration': 180, 'severity': 'High'},
    'thyroid': {'prevalence': 6, 'recovery_rate': 85, 'avg_duration': 365, 'severity': 'Moderate'},
    'cholesterol': {'prevalence': 20, 'recovery_rate': 80, 'avg_duration': 365, 'severity': 'Moderate'},
    'constipation': {'prevalence': 25, 'recovery_rate': 95, 'avg_duration': 5, 'severity': 'Low'},
    'dizziness': {'prevalence': 18, 'recovery_rate': 90, 'avg_duration': 3, 'severity': 'Low'},
    'fatigue': {'prevalence': 35, 'recovery_rate': 85, 'avg_duration': 14, 'severity': 'Low'},
    'default': {'prevalence': 25, 'recovery_rate': 88, 'avg_duration': 7, 'severity': 'Moderate'}
}

def get_disease_stats(symptom):
    symptom_lower = symptom.lower()
    for key in DISEASE_STATS:
        if key in symptom_lower:
            return DISEASE_STATS[key]
    return DISEASE_STATS['default']

def create_analytics_graph(symptom):
    """Create Plotly gauge charts for disease analytics"""
    stats = get_disease_stats(symptom)
    
    fig = go.Figure()
    
    # Recovery Rate Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=stats['recovery_rate'],
        title={'text': "Recovery Rate %", 'font': {'size': 16, 'color': '#00695C'}},
        delta={'reference': 85, 'increasing': {'color': "#00695C"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#00695C"},
            'bar': {'color': "#00695C"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#B2DFDB",
            'steps': [
                {'range': [0, 50], 'color': '#FFCDD2'},
                {'range': [50, 75], 'color': '#FFE0B2'},
                {'range': [75, 100], 'color': '#C8E6C9'}
            ],
            'threshold': {'line': {'color': "#E53935", 'width': 4}, 'thickness': 0.75, 'value': stats['recovery_rate']}
        },
        domain={'x': [0, 0.45], 'y': [0, 1]}
    ))
    
    # Prevalence Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=stats['prevalence'],
        title={'text': "Prevalence %", 'font': {'size': 16, 'color': '#00695C'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': "#4DB6AC"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#B2DFDB",
            'steps': [
                {'range': [0, 20], 'color': '#C8E6C9'},
                {'range': [20, 50], 'color': '#FFE0B2'},
                {'range': [50, 100], 'color': '#FFCDD2'}
            ]
        },
        domain={'x': [0.55, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(255,255,255,0.9)',
        font={'color': "#00695C", 'family': "Inter"},
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig, stats

# GEMINI AI INTEGRATION
GEMINI_API_KEY = "AIzaSyDPT3BMbLEUj17haABGGpSsx70lDoPUgEA"

def get_gemini_health_advice(symptom, medicines):
    """Get AI health advice from Google Gemini API"""
    try:
        medicine_list = ", ".join([m.get('Medicine Name', '') for m in medicines[:5]])
        
        prompt = f"""You are a helpful medical AI assistant. The user has symptoms: "{symptom}".
        Based on our database, we found these medicines: {medicine_list}.
        
        Please provide:
        1. Brief health tip (1-2 sentences)
        2. When to see a doctor
        3. Home remedies (if applicable)
        
        Keep response under 100 words. Be professional and caring."""
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        
        data = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 150}
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
            if 'candidates' in result and result['candidates']:
                return result['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        print(f"Gemini API error: {e}")
    
    return None

# GLOBAL MODEL STATE
df1, df2 = None, None
vectorizer, tfidf_matrix = None, None
DATA_LOADED = False

def initialize_model():
    """Initialize ML model at startup"""
    global df1, df2, vectorizer, tfidf_matrix, DATA_LOADED
    
    print("=" * 50)
    print("   MEDIMATCH ML ENGINE - Initializing")
    print("=" * 50)
    
    df1, df2 = load_data()
    
    if df1 is not None:
        print(f"✅ Dataset 1: {len(df1):,} medicines")
        if df2 is not None:
            print(f"✅ Dataset 2: {len(df2):,} medicines")
        
        vectorizer, tfidf_matrix = train_model(df1)
        DATA_LOADED = True
        print("✅ ML Engine Ready!")
    else:
        DATA_LOADED = False
        print("❌ Failed to load datasets")
    
    print("=" * 50)
    return DATA_LOADED

def get_recommendations(user_input, top_n=15):
    """High-level API to get medicine recommendations with smart validation"""
    if not DATA_LOADED:
        return [], "System not initialized. Please try again."
    
    # Smart validation - reject non-medical queries
    is_valid, error_msg = is_medical_query(user_input)
    if not is_valid:
        return [], error_msg
    
    results = search_medicines(user_input, df1, vectorizer, tfidf_matrix, top_n)
    
    if not results:
        return [], "No medicines found. Try describing your symptoms differently."
    
    return results, None

# TEST
if __name__ == '__main__':
    success = initialize_model()
    
    if success:
        print("\n🔍 Testing 'headache'...")
        results = get_recommendations("headache", top_n=5)
        
        for i, med in enumerate(results, 1):
            print(f"   {i}. {med['Medicine Name']} - {med['Match Score']}")
        
        print("\n📊 Analytics...")
        fig, stats = create_analytics_graph("headache")
        print(f"   Recovery: {stats['recovery_rate']}%, Severity: {stats['severity']}")
        
        print("\n✅ ML ENGINE TEST COMPLETE")
