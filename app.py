import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
from dotenv import load_dotenv
from collections import Counter
import re
from dataclasses import dataclass, field
from typing import List, Optional
from groq import Groq
import tempfile
import librosa
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import cv2
import io
import json
import hashlib
from pathlib import Path

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

# ==================== PROPER .ENV LOADING ====================
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# ==================== STREAMLIT SECRETS ====================
try:
    SYSTEM_API_KEY = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    SYSTEM_API_KEY = os.getenv("GROQ_API_KEY")

# ==================== MOBILE-FRIENDLY CSS 📱 ====================

def inject_mobile_css():
    """Inject mobile-responsive CSS"""
    st.markdown("""
    <style>
    /* Mobile Optimization */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem !important;
            max-width: 100% !important;
        }
        
        h1 {
            font-size: 1.8rem !important;
        }
        
        h2 {
            font-size: 1.4rem !important;
        }
        
        h3 {
            font-size: 1.2rem !important;
        }
        
        .stButton button {
            width: 100% !important;
            padding: 0.75rem !important;
            font-size: 1.1rem !important;
        }
        
        .stTextArea textarea {
            font-size: 1rem !important;
        }
        
        .stSelectbox select {
            font-size: 1rem !important;
        }
        
        div[data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
        }
        
        div[data-testid="metric-container"] {
            padding: 0.5rem !important;
        }
        
        .streamlit-expanderHeader {
            font-size: 1.1rem !important;
            padding: 1rem !important;
        }
        
        section[data-testid="stSidebar"] {
            width: 80vw !important;
        }
    }
    
    .stButton button {
        min-height: 3rem;
        touch-action: manipulation;
    }
    
    .stTextInput input, .stTextArea textarea {
        font-size: 16px !important;
    }
    
    img, video {
        max-width: 100% !important;
        height: auto !important;
    }
    
    @media (max-width: 768px) {
        button[title="View fullscreen"] {
            display: none;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== USAGE ANALYTICS 📊 ====================

def init_analytics():
    """Initialize analytics tracking"""
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {
            'session_id': generate_session_id(),
            'page_views': 0,
            'analyses_completed': 0,
            'questions_generated': 0,
            'templates_used': [],
            'languages_used': [],
            'avg_analysis_time': [],
            'device_type': detect_device_type(),
            'session_start': datetime.now().isoformat(),
            'user_events': []
        }

def generate_session_id():
    """Generate unique session ID"""
    timestamp = datetime.now().isoformat()
    random_data = str(time.time())
    return hashlib.md5(f"{timestamp}{random_data}".encode()).hexdigest()[:12]

def detect_device_type():
    """Detect if user is on mobile/tablet/desktop"""
    return "unknown"

def track_event(event_name: str, properties: dict = None):
    """Track user events"""
    if 'analytics' not in st.session_state:
        init_analytics()
    
    event = {
        'timestamp': datetime.now().isoformat(),
        'event': event_name,
        'properties': properties or {}
    }
    
    st.session_state.analytics['user_events'].append(event)
    
    if event_name == 'analysis_completed':
        st.session_state.analytics['analyses_completed'] += 1
        if 'time' in properties:
            st.session_state.analytics['avg_analysis_time'].append(properties['time'])
    elif event_name == 'questions_generated':
        st.session_state.analytics['questions_generated'] += 1
    elif event_name == 'template_selected':
        if properties and 'template' in properties:
            st.session_state.analytics['templates_used'].append(properties['template'])
    elif event_name == 'language_selected':
        if properties and 'language' in properties:
            st.session_state.analytics['languages_used'].append(properties['language'])

def get_analytics_summary():
    """Get analytics summary"""
    if 'analytics' not in st.session_state:
        return {}
    
    analytics = st.session_state.analytics
    avg_time = 0
    if analytics['avg_analysis_time']:
        avg_time = sum(analytics['avg_analysis_time']) / len(analytics['avg_analysis_time'])
    
    return {
        'session_id': analytics['session_id'],
        'device_type': analytics['device_type'],
        'session_duration': (datetime.now() - datetime.fromisoformat(analytics['session_start'])).total_seconds(),
        'analyses_completed': analytics['analyses_completed'],
        'questions_generated': analytics['questions_generated'],
        'avg_analysis_time': round(avg_time, 2),
        'most_used_template': max(set(analytics['templates_used']), key=analytics['templates_used'].count) if analytics['templates_used'] else None,
        'total_events': len(analytics['user_events'])
    }

def export_analytics():
    """Export analytics as JSON"""
    if 'analytics' not in st.session_state:
        return "{}"
    return json.dumps(st.session_state.analytics, indent=2)

# ==================== API KEY MANAGEMENT ====================

@st.cache_resource
def _cached_groq_client(api_key: str):
    """Create and cache a Groq client for a given API key."""
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to create Groq client: {str(e)}")
        return None

# Load Haar cascades once
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
SMILE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def get_groq_client():
    """Get Groq client - user's key preferred, fallback to system key"""
    user_key = st.session_state.get('user_api_key')
    if user_key:
        try:
            return _cached_groq_client(user_key)
        except Exception as e:
            st.error(f"❌ Invalid user API key: {str(e)}")
            return None
    
    if SYSTEM_API_KEY:
        try:
            return _cached_groq_client(SYSTEM_API_KEY)
        except Exception as e:
            st.warning(f"⚠️ System API key error: {str(e)}")
            return None
    
    return None

# ==================== CORRECT GROQ MODEL ====================
GROQ_MODEL = "openai/gpt-oss-120b"

# ==================== USAGE TRACKING ====================

def init_usage_tracking():
    """Track free tier usage"""
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0
    if 'last_reset' not in st.session_state:
        st.session_state.last_reset = datetime.now().date()
    
    if st.session_state.last_reset != datetime.now().date():
        st.session_state.usage_count = 0
        st.session_state.last_reset = datetime.now().date()

def check_usage_limit():
    """Check if user exceeded free tier"""
    FREE_TIER_LIMIT = 5
    
    if 'user_api_key' in st.session_state and st.session_state.user_api_key:
        return True
    
    if st.session_state.usage_count >= FREE_TIER_LIMIT:
        return False
    
    return True

def increment_usage():
    """Increment usage counter"""
    if 'user_api_key' not in st.session_state or not st.session_state.user_api_key:
        st.session_state.usage_count += 1

# ==================== INDUSTRY TEMPLATES ====================

INDUSTRY_TEMPLATES = {
    "Software Engineering": {
        "icon": "💻",
        "keywords": ["Python", "JavaScript", "System Design", "Algorithms", "Cloud", "Docker", "Kubernetes", "API", "Database", "Git"],
        "sample_jd": """We are seeking a Senior Software Engineer to join our team.

Requirements:
- 5+ years experience with Python, JavaScript, or Java
- Strong knowledge of system design and algorithms
- Experience with cloud platforms (AWS/GCP/Azure)
- Proficiency in Docker, Kubernetes, and microservices
- Excellent problem-solving and communication skills

Responsibilities:
- Design and implement scalable backend systems
- Collaborate with cross-functional teams
- Write clean, maintainable code
- Conduct code reviews and mentor junior developers"""
    },
    "Product Management": {
        "icon": "📊",
        "keywords": ["Product Strategy", "Roadmap", "Stakeholders", "Agile", "User Research", "KPIs", "A/B Testing", "Analytics"],
        "sample_jd": """Seeking an experienced Product Manager to drive product strategy.

Requirements:
- 3+ years in product management
- Strong understanding of Agile methodologies
- Experience with user research and data analytics
- Excellent stakeholder management skills

Responsibilities:
- Define product vision and roadmap
- Conduct user research and competitive analysis
- Work with engineering and design teams
- Track KPIs and make data-driven decisions"""
    },
    "Data Science": {
        "icon": "📈",
        "keywords": ["Python", "Machine Learning", "SQL", "Statistics", "TensorFlow", "PyTorch", "Data Visualization"],
        "sample_jd": """Looking for a Data Scientist to build predictive models.

Requirements:
- 3+ years in data science or analytics
- Strong Python and SQL skills
- Experience with ML frameworks
- Knowledge of statistical analysis

Responsibilities:
- Build and deploy machine learning models
- Analyze large datasets to extract insights
- Collaborate with engineering teams
- Present findings to stakeholders"""
    }
}

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Hindi": "hi",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko"
}

# ==================== DATA MODELS ====================

@dataclass
class PhotoAnalysis:
    smile_detected: bool = False
    face_detected: bool = False
    confidence: str = "Unknown"
    posture: str = "Unknown"
    eye_contact: str = "Unknown"

@dataclass
class VoiceAnalysis:
    confidence_score: float = 0.0
    avg_volume: float = 0.0
    duration_seconds: float = 0.0
    speaking_pace: float = 0.0
    pitch_variation: float = 0.0
    live_feedback: List[str] = field(default_factory=list)

@dataclass
class Answer:
    q_idx: int
    question: str
    text: str
    voice_analysis: Optional[VoiceAnalysis] = None
    photo_analysis: Optional[PhotoAnalysis] = None
    term_analysis: dict = field(default_factory=dict)
    relevance_score: float = 0.0
    ai_feedback: str = ""
    improved_answer: str = ""
    ideal_answer: str = ""
    similarity_score: float = 0.0

@dataclass
class Session:
    jd: str = ""
    jd_keywords: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    answers: List[Answer] = field(default_factory=list)
    total_times: List[float] = field(default_factory=list)
    industry: str = ""
    language: str = "en"

# ==================== AUDIO PROCESSING ====================

def transcribe_audio(audio_bytes: bytes, language: str = "en") -> str:
    """Multi-language transcription"""
    groq_client = get_groq_client()
    if not groq_client:
        return "[Error: No API key available]"
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, "rb") as audio_file:
                transcription = groq_client.audio.transcriptions.create(
                    file=(tmp_file_path, audio_file.read()),
                    model="whisper-large-v3-turbo",
                    response_format="text"
                )
                
            if transcription:
                return transcription.strip()
            else:
                return "[Error: Empty transcription]"
        
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Transcription failed: {error_msg}")
        return f"[Error: {error_msg}]"

def analyze_voice_confidence_fast(audio_bytes: bytes, transcribed_text: str) -> VoiceAnalysis:
    """Voice analysis with real-time feedback"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        y, sr = librosa.load(tmp_file_path, sr=8000, mono=True)
        
        rms = librosa.feature.rms(y=y)[0]
        avg_volume = float(np.mean(rms))
        
        duration = len(y) / sr
        word_count = len(transcribed_text.split())
        speaking_pace = (word_count / duration) * 60 if duration > 0 else 0
        
        pitch_values = librosa.yin(y, fmin=75, fmax=300, sr=sr)
        pitch_std = float(np.std(pitch_values[pitch_values > 0])) if len(pitch_values[pitch_values > 0]) > 0 else 0.0
        
        volume_score = min(100, avg_volume * 15000)
        pace_score = 100 if 120 <= speaking_pace <= 150 else 70 if 100 <= speaking_pace < 180 else 40
        pitch_score = min(100, pitch_std * 3)
        
        confidence_score = volume_score * 0.4 + pace_score * 0.3 + pitch_score * 0.3
        
        live_feedback = []
        if avg_volume < 0.01:
            live_feedback.append("🔊 Speak louder")
        if speaking_pace < 100:
            live_feedback.append("⏩ Speed up")
        elif speaking_pace > 180:
            live_feedback.append("⏸️ Slow down")
        if pitch_std < 20:
            live_feedback.append("🎵 Add vocal variety")
        if confidence_score >= 80:
            live_feedback.append("✨ Excellent!")
        
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        return VoiceAnalysis(
            confidence_score=round(confidence_score, 1),
            avg_volume=round(avg_volume, 4),
            duration_seconds=round(duration, 1),
            speaking_pace=round(speaking_pace, 1),
            pitch_variation=round(pitch_std, 2),
            live_feedback=live_feedback
        )
    except Exception as e:
        st.error(f"Voice analysis error: {str(e)}")
        return VoiceAnalysis()

# ==================== PHOTO ANALYSIS ====================

def analyze_photo_ultra_fast(image_bytes) -> PhotoAnalysis:
    """Lightning-fast photo analysis"""
    try:
        if image_bytes is None:
            return PhotoAnalysis(face_detected=False, confidence="No image")
            
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return PhotoAnalysis(face_detected=False, confidence="Invalid image")
        
        img = cv2.resize(img, (320, 240))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 3)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            
            eyes = EYE_CASCADE.detectMultiScale(roi_gray, 1.3, 3)
            smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 1.8, 15)
            
            smile_detected = len(smiles) > 0
            eye_contact = "Good ✅" if len(eyes) >= 2 else "Improve ⚠️"
            
            img_center_x = img.shape[1] // 2
            face_center_x = x + w // 2
            centered = abs(face_center_x - img_center_x) < img.shape[1] * 0.25
            
            return PhotoAnalysis(
                smile_detected=smile_detected,
                face_detected=True,
                confidence="High" if len(eyes) >= 2 else "Medium",
                posture="Centered ✅" if centered else "Off-center ⚠️",
                eye_contact=eye_contact
            )
        else:
            return PhotoAnalysis(face_detected=False, confidence="No face detected")
    except Exception as e:
        st.error(f"Photo analysis error: {str(e)}")
        return PhotoAnalysis(face_detected=False, confidence="Error")

# ==================== CACHING ====================

@st.cache_data(ttl=3600)
def get_ideal_answer_cached(question: str, keywords_tuple: tuple) -> str:
    """Cache ideal answers - avoids repeated API calls for same question"""
    groq_client = get_groq_client()
    if not groq_client:
        return ""
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Provide a perfect interview answer (80-100 words)."},
                {"role": "user", "content": f"Question: {question}\nKeywords: {', '.join(keywords_tuple)}"}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Could not generate ideal answer: {str(e)}")
        return ""

# ==================== OPTIMIZED LLM FUNCTIONS ====================

# ✅ NEW: Combined feedback + improved in one call
def generate_feedback_and_improved(question: str, answer_text: str, jd_keywords: List[str], 
                                   voice_analysis: Optional[VoiceAnalysis] = None,
                                   photo_analysis: Optional[PhotoAnalysis] = None) -> tuple:
    """Combined LLM call: Gets both feedback AND improved answer in ONE request (faster!)"""
    groq_client = get_groq_client()
    if not groq_client:
        return "", ""
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system", 
                    "content": """Provide two things:
1. FEEDBACK: Concise feedback on the answer (2-3 sentences max)
2. IMPROVED: How to improve this answer (2-3 sentences max)

Format your response as:
FEEDBACK: [your feedback here]
IMPROVED: [improvement suggestions here]"""
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}\n\nAnswer: {answer_text}\n\nKeywords: {', '.join(jd_keywords)}"
                }
            ],
            temperature=0.5,
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse the response
        feedback = ""
        improved = ""
        
        if "FEEDBACK:" in response_text and "IMPROVED:" in response_text:
            parts = response_text.split("IMPROVED:")
            feedback = parts[0].replace("FEEDBACK:", "").strip()
            improved = parts[1].strip()
        else:
            # Fallback if format is different
            feedback = response_text[:100]
            improved = response_text[100:]
        
        return feedback, improved
        
    except Exception as e:
        return f"Error: {str(e)}", ""

def generate_questions(jd_text: str, industry: str = "") -> List[str]:
    """Generate questions"""
    groq_client = get_groq_client()
    if not groq_client:
        return []
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": f"Generate 5 interview questions. One per line."},
                {"role": "user", "content": f"Job: {jd_text}"}
            ],
            temperature=0.7,
            max_tokens=400
        )
        
        questions = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
        questions = [re.sub(r'^[\d\.\)\-]+\s*', '', q) for q in questions]
        return [q for q in questions if len(q) > 10][:5]
    except Exception as e:
        st.error(f"Failed to generate questions: {str(e)}")
        return []

def extract_jd_keywords(jd_text: str) -> List[str]:
    """Extract keywords"""
    groq_client = get_groq_client()
    if not groq_client:
        return []
    
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Extract 10 key skills as comma-separated list."},
                {"role": "user", "content": jd_text}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        keywords = [k.strip() for k in response.choices[0].message.content.strip().split(',')]
        return keywords[:10]
    except Exception as e:
        st.error(f"Failed to extract keywords: {str(e)}")
        return []

# ==================== TEXT ANALYSIS ====================

def analyze_terms(answer_text: str) -> dict:
    """Text analysis"""
    fillers = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
    words = re.findall(r'\b\w+\b', answer_text.lower())
    word_freq = Counter(words)
    filler_count = sum(word_freq.get(f.replace(' ', ''), 0) for f in fillers)
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
                  'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had', 'i', 'you', 'my', 'your'}
    
    meaningful = {w: c for w, c in word_freq.items() if w not in stop_words and len(w) > 2}
    top_terms = Counter(meaningful).most_common(5)
    
    return {
        'total_words': len(words),
        'unique_words': len(word_freq),
        'filler_count': filler_count,
        'top_terms': top_terms
    }

def calculate_relevance_score(answer_text: str, jd_keywords: List[str]) -> float:
    """Relevance score"""
    if not jd_keywords:
        return 0.0
    answer_lower = answer_text.lower()
    matches = sum(1 for kw in jd_keywords if kw.lower() in answer_lower)
    return round((matches / len(jd_keywords)) * 100, 1)

def calculate_similarity_score(user_answer: str, ideal_answer: str) -> float:
    """Calculate similarity"""
    user_words = set(re.findall(r'\b\w+\b', user_answer.lower()))
    ideal_words = set(re.findall(r'\b\w+\b', ideal_answer.lower()))
    
    if not ideal_words:
        return 0.0
    
    overlap = len(user_words & ideal_words)
    similarity = (overlap / len(ideal_words)) * 100
    
    return round(min(similarity, 100), 1)

# ==================== ULTRA-FAST ANALYSIS ====================
# ✅ NEW: Parallel API calls with combined prompts
def analyze_answer_ultra_fast(question: str, answer_text: str, jd_keywords: List[str], 
                              voice_analysis: Optional[VoiceAnalysis],
                              photo_analysis: Optional[PhotoAnalysis] = None) -> tuple:
    """Ultra-optimized with parallel API calls and combined prompts (12-15 seconds total)"""
    
    placeholder = st.empty()
    
    # STEP 1: Local analysis (parallel - instant)
    placeholder.info("📊 Analyzing your response...")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_terms = executor.submit(analyze_terms, answer_text)
        f_relevance = executor.submit(calculate_relevance_score, answer_text, jd_keywords)
        
        terms = f_terms.result()
        relevance = f_relevance.result()
    
    placeholder.success("✅ Initial analysis done!")
    time.sleep(0.1)  # Minimal delay
    
    # STEP 2: Parallel AI calls (combined + ideal)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Combined call (feedback + improved in ONE API request)
            f_combined = executor.submit(
                generate_feedback_and_improved, 
                question, answer_text, jd_keywords, 
                voice_analysis, photo_analysis
            )
            
            # Ideal answer (with caching - much faster on repeats)
            f_ideal = executor.submit(
                get_ideal_answer_cached, 
                question, 
                tuple(jd_keywords)  # Convert to tuple for caching
            )
            
            placeholder.info("🤖 Analyzing with AI... (parallel processing)")
            
            # Get results as they complete
            feedback, improved = f_combined.result()
            ideal = f_ideal.result()
        
        placeholder.empty()
        
    except Exception as e:
        placeholder.error(f"❌ Analysis failed: {str(e)}")
        return {}, 0.0, f"Error: {str(e)}", "", "", 0.0
    
    similarity = calculate_similarity_score(answer_text, ideal)
    
    return terms, relevance, feedback, improved, ideal, similarity

# ==================== PDF GENERATION ====================

def generate_pdf_report(session: Session) -> bytes:
    """Generate PDF report"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        story = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24,
                                     textColor=colors.HexColor('#1f77b4'), spaceAfter=30, alignment=TA_CENTER)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16,
                                       textColor=colors.HexColor('#2ca02c'), spaceAfter=12, spaceBefore=12)
        
        story.append(Paragraph("Interview Report", title_style))
        story.append(Paragraph(f"{datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        if session.jd_keywords:
            story.append(Paragraph("Key Skills", heading_style))
            story.append(Paragraph(", ".join(session.jd_keywords), styles['Normal']))
            story.append(Spacer(1, 0.3*inch))
        
        for answer in session.answers:
            story.append(PageBreak())
            story.append(Paragraph(f"Question {answer.q_idx}", heading_style))
            story.append(Paragraph(answer.question, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            story.append(Paragraph("Your Answer", heading_style))
            story.append(Paragraph(answer.text, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            if answer.ai_feedback:
                story.append(Paragraph("AI Feedback", heading_style))
                story.append(Paragraph(answer.ai_feedback, styles['Normal']))
        
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return b""

# ==================== MAIN UI ====================

def main():
    st.set_page_config(
        page_title="AI Interview Rehearsal Coach",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    inject_mobile_css()
    
    if 'session' not in st.session_state:
        st.session_state.session = Session()
    
    init_usage_tracking()
    init_analytics()
    
    session = st.session_state.session
    
    track_event('page_view')
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.markdown("### 🔑 API Key")
        
        user_key_input = st.text_input(
            "Groq API Key (Optional)",
            type="password",
            help="Get FREE key: console.groq.com/keys",
            placeholder="Leave empty to use free tier"
        )
        
        if user_key_input:
            st.session_state.user_api_key = user_key_input
            st.success("✅ Using your personal key - Unlimited usage!")
            track_event('api_key_added')
        else:
            st.session_state.user_api_key = None
            
            remaining = max(0, 5 - st.session_state.usage_count)
            
            if SYSTEM_API_KEY:
                st.info(f"🆓 Free tier: {remaining}/5 requests today")
                
                if remaining == 0:
                    st.error("⚠️ Daily limit reached!")
                    st.info("👉 Add your own API key for unlimited usage")
                else:
                    st.success("✅ Using shared free API key")
            else:
                st.error("⚠️ No API key configured")
                st.info("👉 Add your API key to continue")
        
        st.markdown("---")
        
        st.markdown("### 🔗 Links")
        st.markdown("[Get API Key](https://console.groq.com/keys)")
        
        st.markdown("---")
        
        st.markdown("### 📊 Your Stats")
        analytics = get_analytics_summary()
        st.metric("Analyses", analytics.get('analyses_completed', 0))
        if analytics.get('avg_analysis_time', 0) > 0:
            st.metric("Avg Time", f"{analytics['avg_analysis_time']:.1f}s")
        
        if st.checkbox("📥 Export Analytics"):
            st.download_button(
                "Download JSON",
                data=export_analytics(),
                file_name=f"analytics_{analytics['session_id']}.json",
                mime="application/json"
            )
    
    # ==================== MAIN APP ====================
    
    st.title("⚡ AI Interview Rehearsal Coach")
    st.caption("*Mobile-optimized • Ultra-fast analysis (12-15s)*")
    
    groq_client = get_groq_client()
    
    if not groq_client:
        st.error("⚠️ **No API Key Available**")
        
        if SYSTEM_API_KEY:
            st.info("✅ Shared API key is configured but there's an issue.")
            st.error("Please try refreshing the page or contact support.")
        else:
            st.info("👈 Either:")
            st.markdown("""
            1. **Add your own API key** in the sidebar for unlimited usage
            2. **Wait for the owner** to configure a shared API key
            """)
            st.markdown("[Get your free API key](https://console.groq.com/keys)")
        
        with st.expander("ℹ️ How to get your own API key"):
            st.markdown("""
            1. Go to https://console.groq.com/keys
            2. Sign up or log in
            3. Create a new API key
            4. Copy and paste it in the sidebar
            5. You'll get unlimited free usage!
            """)
        st.stop()
    
    if not check_usage_limit():
        st.error("⚠️ **Daily Limit Reached (5 requests)**")
        st.info("👈 Add your own API key in the sidebar for unlimited usage")
        st.stop()
    
    st.markdown("---")
    
    st.header("🎯 Choose Your Path")
    
    tab1, tab2 = st.tabs(["🚀 Templates", "📝 Custom"])
    
    with tab1:
        template_options = list(INDUSTRY_TEMPLATES.keys())
        
        selected_template = st.selectbox(
            "Select industry:",
            options=template_options,
            index=0
        )
        
        template = INDUSTRY_TEMPLATES[selected_template]
        st.success(f"{template['icon']} **{selected_template}**")
        
        with st.expander("📄 View JD"):
            st.text(template['sample_jd'])
        
        if st.button(f"🚀 Use Template", type="primary", use_container_width=True):
            session.jd = template['sample_jd']
            session.jd_keywords = template['keywords']
            session.industry = selected_template
            
            track_event('template_selected', {'template': selected_template})
            
            with st.spinner("Generating..."):
                session.questions = generate_questions(session.jd, selected_template)
                session.answers = []
                track_event('questions_generated', {'count': len(session.questions)})
                if session.questions:
                    st.success(f"✅ {len(session.questions)} questions!")
                    st.rerun()
                else:
                    st.error("Failed to generate questions. Check your API key.")
    
    with tab2:
        selected_language = st.selectbox(
            "🌍 Language:",
            options=list(SUPPORTED_LANGUAGES.keys()),
            index=0
        )
        session.language = SUPPORTED_LANGUAGES[selected_language]
        
        track_event('language_selected', {'language': selected_language})
        
        jd_input = st.text_area(
            "Paste job description:",
            value=session.jd,
            height=200,
            placeholder="Paste JD here..."
        )
        
        if st.button("🚀 Generate", type="primary", use_container_width=True):
            if not jd_input.strip():
                st.error("Paste a JD first!")
            else:
                with st.spinner("Analyzing..."):
                    session.jd = jd_input
                    session.jd_keywords = extract_jd_keywords(jd_input)
                    session.questions = generate_questions(jd_input)
                    session.answers = []
                    track_event('questions_generated', {'count': len(session.questions), 'custom': True})
                    if session.questions:
                        st.success(f"✅ {len(session.questions)} questions!")
                        st.rerun()
                    else:
                        st.error("Failed to generate questions.")
    
    # ==================== QUESTIONS ====================
    
    if session.questions:
        st.markdown("---")
        st.header("❓ Answer Questions")
        
        if session.jd_keywords:
            st.caption(f"🎯 Key Skills: {', '.join(session.jd_keywords[:5])}")
        
        for idx, question in enumerate(session.questions, 1):
            with st.expander(f"**Q{idx}:** {question[:50]}...", expanded=(idx==1)):
                st.markdown(f"### {question}")
                
                st.markdown("#### 📸 Photo")
                camera_photo = st.camera_input("Take photo", key=f"cam_{idx}", label_visibility="collapsed")
                
                st.markdown("#### 🎤 Audio")
                audio_bytes = audio_recorder(
                    text="", recording_color="#e74c3c", neutral_color="#3498db",
                    icon_name="microphone", icon_size="2x", pause_threshold=3.0,
                    sample_rate=16000, key=f"audio_{idx}"
                )
                
                if audio_bytes and camera_photo:
                    st.success("✅ Captured!")
                    
                    if st.button(f"⚡ Analyze", key=f"btn_{idx}", type="primary", use_container_width=True):
                        
                        if not check_usage_limit():
                            st.error("⚠️ Daily limit reached! Add your own API key for unlimited usage.")
                            st.stop()
                        
                        start = time.time()
                        track_event('analysis_started', {'question_idx': idx})
                        
                        with st.spinner("⚡ Processing audio & photo..."):
                            photo_bytes = camera_photo.getvalue()
                            
                            # Transcribe audio
                            transcribed_text = transcribe_audio(audio_bytes, session.language)
                            
                            # Check for transcription errors
                            if "[Error:" in transcribed_text:
                                st.error(f"❌ {transcribed_text}")
                                st.stop()
                            
                            # Analyze photo and voice
                            photo_analysis = analyze_photo_ultra_fast(photo_bytes)
                            voice_analysis = analyze_voice_confidence_fast(audio_bytes, transcribed_text)
                            
                            # Show interim results
                            st.info(f"📝 Transcribed: {transcribed_text[:100]}...")
                            
                            # Run ULTRA-FAST analysis with parallel calls
                            term_analysis, relevance_score, ai_feedback, improved_answer, ideal_answer, similarity = analyze_answer_ultra_fast(
                                question, transcribed_text, session.jd_keywords, voice_analysis, photo_analysis
                            )
                            
                            total_time = time.time() - start
                            
                            answer_obj = Answer(
                                q_idx=idx, question=question, text=transcribed_text,
                                voice_analysis=voice_analysis, photo_analysis=photo_analysis,
                                term_analysis=term_analysis, relevance_score=relevance_score,
                                ai_feedback=ai_feedback, improved_answer=improved_answer,
                                ideal_answer=ideal_answer, similarity_score=similarity
                            )
                            
                            session.answers = [a for a in session.answers if a.q_idx != idx]
                            session.answers.append(answer_obj)
                            session.total_times.append(total_time)
                            
                            increment_usage()
                            track_event('analysis_completed', {
                                'question_idx': idx,
                                'time': total_time,
                                'relevance_score': relevance_score,
                                'similarity_score': similarity
                            })
                            
                            st.success(f"✅ Analysis complete in {total_time:.1f}s! (Ultra-fast ⚡)")
                            st.rerun()
                
                existing = next((a for a in session.answers if a.q_idx == idx), None)
                if existing:
                    st.markdown("---")
                    st.markdown("## 📊 Results")
                    
                    with st.expander("📝 Your Answer", expanded=False):
                        st.write(existing.text)
                    
                    if existing.ideal_answer:
                        with st.expander("⭐ Ideal Answer"):
                            st.write(existing.ideal_answer)
                            st.metric("Similarity", f"{existing.similarity_score}%")
                    
                    st.markdown("### 📈 Scores")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if existing.voice_analysis:
                            st.metric("Voice", f"{existing.voice_analysis.confidence_score:.0f}/100")
                        st.metric("Words", existing.term_analysis.get('total_words', 0))
                    
                    with col2:
                        if existing.photo_analysis and existing.photo_analysis.face_detected:
                            st.metric("Smile", "✅" if existing.photo_analysis.smile_detected else "❌")
                        st.metric("Relevance", f"{existing.relevance_score}%")
                    
                    st.markdown("### 🤖 Feedback")
                    st.info(existing.ai_feedback)
                    
                    if existing.improved_answer:
                        with st.expander("✨ Improved Version"):
                            st.success(existing.improved_answer)
        
        if session.answers:
            st.markdown("---")
            st.header("📄 Report")
            
            if st.button("📥 Generate PDF", type="primary", use_container_width=True):
                track_event('pdf_generated', {'answers_count': len(session.answers)})
                with st.spinner("Creating..."):
                    pdf = generate_pdf_report(session)
                    st.download_button(
                        "⬇️ Download",
                        data=pdf,
                        file_name=f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Ready!")
    else:
        if session.jd:
            st.info("📋 No questions generated yet. Try again or use a different JD.")
        else:
            st.info("👆 Start by selecting a template or pasting a job description.")
    
    st.markdown("---")
    st.caption("⚡ AI Interview Rehearsal Coach - Ultra-fast analysis powered by Groq")

if __name__ == "__main__":
    main()