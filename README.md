# 🎤 AI Interview Rehearsal Coach

> Practice smarter. Speak confidently. Get hired.

---

## 🚀 Overview

**AI Interview Rehearsal Coach** is your personal AI-powered interview companion.  
It helps you **rehearse interviews**, analyze your **answers and voice tone**, and build genuine confidence — not just memorize responses.

Paste a **job description (JD)** or pick from **industry templates**, answer AI-generated questions (by voice or text), and receive instant, intelligent feedback on:

- ✅ Relevance & keyword coverage
- 🧠 Structure & clarity of your answer
- 🎧 Voice confidence, tone, and pacing
- 💡 AI-suggested improvements + ideal answers
- 📄 Downloadable interview report in PDF format

---

## 🧠 Key Features

| Category                      | Features                                            |
| ----------------------------- | --------------------------------------------------- |
| **Smart Questioning**         | Tailored interview questions generated from your JD |
| **Voice Confidence Feedback** | Analyzes tone, fluency, and vocal clarity           |
| **Answer Analysis**           | Rates structure, relevance, and JD alignment        |
| **Industry Templates**        | Ready-to-use JDs for common roles                   |
| **PDF Report**                | AI-powered summary of your performance              |
| **Session Analytics**         | Tracks improvement and usage                        |
| **Mobile Optimized**          | Clean, touch-friendly Streamlit interface           |

---

## 🧩 Tech Stack

| Layer            | Technology                              |
| ---------------- | --------------------------------------- |
| **Frontend**     | [Streamlit](https://streamlit.io/)      |
| **AI/LLM**       | [Groq API](https://groq.com/)           |
| **Audio**        | `audio_recorder_streamlit`              |
| **Data & Logic** | Python `dataclasses`, `asyncio`, `json` |
| **PDF Export**   | `reportlab`                             |
| **Analytics**    | Custom session tracking                 |
| **Styling**      | Responsive CSS for mobile-first UI      |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-interview-rehearsal-coach.git
cd ai-interview-rehearsal-coach
```

### 2. Setup the Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install requirements.txt

```bash
uv pip install -r requirements.txt
```

### 4. Create dotenv file and paste your key

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the cmd

```bash
streamlit run app.py
```

## 📊 Output Example

After each practice session, you’ll receive:

- 🧐 **Personalized AI feedback** on each answer
- 🗝️ **Keyword & relevance score**
- 🎙️ **Voice tone and confidence breakdown**
- 💬 **Ideal and improved answers**
- 📄 **PDF Interview Report** summarizing your performance

---

## 🧭 Future Enhancements

- 🎥 **Live coaching dashboard** (real-time voice + posture metrics)
- 📈 **Progress tracking and analytics dashboard**
- 🌍 **Multi-language support**
- 🔗 **LinkedIn / Drive integration**

## 👨‍💻 Author

**Anuj P.**  
AI x Productivity Enthusiast • Builder of calm, focused AI tools that help people learn, prepare, and grow.

🌐 **Connect with me**

- [💼 LinkedIn](https://www.linkedin.com/in/anujp24)
- [🐦 Twitter / X](https://twitter.com/anujp24)
- [📰 Substack](https://anujp24.substack.com)
- [🧠 Hashnode](https://anujp24.hashnode.dev)
