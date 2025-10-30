# 🎤 AI Interview Rehearsal Coach

> **Practice smarter. Speak confidently. Get hired.**

Master your interviews with AI-powered real-time feedback on answers, voice tone, and body language.

---

## 🎯 What This Does

Get honest, instant feedback on your interview performance—like having a professional coach in your pocket.

**You:**

- 📝 Paste a job description (or pick a template)
- 🎤 Record your answer to AI-generated questions
- 📸 Take a quick selfie (optional)

**You Get:**

- ✅ Relevance score (does your answer match the job?)
- 🎙️ Voice analysis (confidence, pace, clarity)
- 📸 Body language insights (smile, posture, eye contact)
- 💡 Specific improvement suggestions
- ⭐ Perfect answer example to learn from
- 📄 Professional PDF report

---

## 💡 Key Insights (Why This Works)

### The Problem It Solves

- Most people practice interviews in a vacuum—no feedback
- Professional coaches are expensive ($100-300/session)
- Mock interviews with friends feel awkward and unstructured
- Hard to track progress over multiple practice sessions

### The Solution

- **Instant AI feedback:** No waiting for a callback from a coach
- **Judgment-free:** Practice as many times as you want
- **Quantified improvements:** Track your scores across sessions
- **Comprehensive:** Covers answer quality + delivery (not just one)

### How It's Different

| Aspect         | Traditional Coach | This Tool                 |
| -------------- | ----------------- | ------------------------- |
| Cost           | $100-300/session  | Free (with API key)       |
| Availability   | Limited hours     | 24/7                      |
| Feedback Speed | Days later        | Instant                   |
| Practice Limit | Few sessions      | Unlimited                 |
| Feedback Type  | Subjective        | Data-driven + AI insights |

---

## 🚀 Quick Start (2 Minutes)

### 1. **Get Your Free API Key** (30 seconds)

```
1. Go to https://console.groq.com/keys
2. Sign up (no credit card needed)
3. Copy your API key
```

### 2. **Run the App**

```bash
# Clone & setup
git clone <repo>
cd ai-interview-rehearsal-coach
python3 -m venv venv
source venv/bin/activate

# Install (cv2 MUST be first!)
pip install opencv-python-headless==4.11.0.80
pip install -r requirements.txt

# Create .env
echo "GROQ_API_KEY=your_key_here" > .env

# Run
streamlit run app.py
```

Visit: **http://localhost:8501**

### 3. **Practice**

- Pick a template or paste your job description
- Answer 5 AI-generated questions
- Get feedback on each
- Download your report

---

## 📊 What You Learn

### On Your Answers

- **Relevance Score (0-100):** How well you addressed the job description
- **Keyword Coverage:** Which skills you mentioned vs. what the job needs
- **Structure:** Is your answer organized? (Problem → Solution → Result?)
- **Missing Points:** What else should you mention?

### On Your Delivery

- **Confidence Score (0-100):** Overall impression from tone/pace/volume
- **Speaking Pace:** Words per minute (fast = nervous, slow = unsure)
- **Volume Consistency:** Are you trailing off? Speaking too quietly?
- **Pitch Variation:** Monotone vs. natural, engaging tone
- **Specific Fixes:** "Slow down," "Speak louder," "Add more energy"

### On Your Body Language

- ✅/❌ Face detected (camera working?)
- ✅/❌ Smile detected (engaging?)
- ✅/❌ Eye contact (looking at camera)
- 📍 Posture (centered, not slouching)

---

## 🧠 Real Example

**Question:** "Tell me about a time you solved a difficult technical problem."

**Your Answer (60 seconds):** _"Um, well, I was working on this feature and it broke. I debugged it. It took like, two days but I fixed it."_

**Feedback You Get:**

🔴 **Relevance: 35/100**

- Missing: Problem impact, tools used, what you learned
- Keywords missing: "troubleshooting," "optimization," "documentation"

🎙️ **Confidence: 42/100**

- ❌ Speaking pace: 95 wpm (too fast, sounds nervous)
- ❌ Volume: Quiet and trailing off at the end
- ❌ Pitch: Monotone, no energy

📸 **Body Language:**

- ✅ Face detected
- ❌ No smile detected
- ❌ Looking down (not at camera)

💡 **How to Improve:**
_"Add the business impact. What broke? How did it affect users/revenue? Show your debugging process step-by-step. Mention what you learned and how you applied it later. Slow down, speak from your diaphragm, and maintain eye contact with the camera."_

⭐ **Better Answer:**
_"I was fixing a payment processing bug affecting 5% of transactions—worth ~$50k daily impact. I systematically traced the issue through logs, identified a race condition in the checkout flow, implemented a mutex lock, and added monitoring. This cut payment failures by 99.2%. I documented the fix to prevent similar issues and led a team review on async patterns."_

📈 **Similarity to Ideal:** 72% (good foundation, needs delivery polish)

---

## 🎯 Use Cases

**Before a Big Interview**

- Practice 5-10 times with the real job description
- Track confidence scores improving each session
- Identify your weakest talking points
- Refine your delivery (pace, tone, eye contact)

**Interview Prep Program**

- Week 1: Record baseline (before coaching)
- Weeks 2-3: Daily practice sessions
- Week 4: Final mock interview, download report
- Track confidence scores: 40 → 85

**Right Before the Interview**

- Quick 10-minute confidence check
- Get feedback on your opening 30-second pitch
- Verify camera/audio setup
- Calm your nerves with one successful practice

**After You Get the Job**

- Share your report with mentees
- Show what good interview performance looks like
- Help junior engineers prepare

---

## 📈 Key Metrics You Track

Each practice session records:

- **Relevance Score:** How well you answered the question
- **Confidence Score:** Overall delivery quality
- **Speaking Pace:** Words per minute
- **Sentiment:** Positive/negative tone detection
- **Keyword Match:** What job skills you mentioned
- **Session Duration:** How long you practiced
- **Improvement:** Score vs. previous session

Over time, you see:

- ✅ Confidence trending up (40 → 60 → 75 → 85)
- ✅ Speaking pace stabilizing (120 → 105 → 100 wpm)
- ✅ Answer relevance increasing (50 → 65 → 80 → 88)

---

## 🎓 What Makes It Work

### 1. **AI-Generated Questions Tailor to Your Job**

Not generic "Tell me about yourself." Questions pull actual skills from the job description and generate behavioral/technical questions that fit.

### 2. **Multi-Angle Feedback**

Most people focus only on what they say. This covers:

- ✅ What you say (relevance, keywords, structure)
- ✅ How you say it (tone, pace, clarity)
- ✅ How you look (body language, engagement)

### 3. **Comparison Learning**

See the "ideal answer" right after yours. Not told to memorize it, but to learn the structure, key points, and storytelling approach.

### 4. **Quantified Progress**

Vague feedback ("You did great!") doesn't help. Scores do. Watch your confidence climb from 45 → 88.

### 5. **PDF Reports**

Share your practice results. Proof you prepared. Great for reflecting on your improvement journey.

---

## 🔧 Tech Stack (Why These Tools)

| Tool             | Why                                      | Benefit                          |
| ---------------- | ---------------------------------------- | -------------------------------- |
| **Streamlit**    | Fast web UI, no frontend needed          | Deploy in minutes                |
| **Groq API**     | Lightning-fast LLM (50+ tokens/sec)      | Instant feedback, no waiting     |
| **GPT-OSS-120B** | Strong reasoning, perfect for interviews | Better feedback than GPT-4       |
| **LibROSA**      | Audio analysis                           | Pace, volume, pitch detection    |
| **OpenCV**       | Image processing                         | Smile/face/eye contact detection |
| **ReportLab**    | PDF generation                           | Professional reports             |

---

## 🌟 Features (POC)

### Core Features

- ✅ AI-generated interview questions from job descriptions
- ✅ Real-time voice analysis (confidence, pace, clarity)
- ✅ Photo analysis (smile, face detection, posture)
- ✅ Answer relevance scoring (0-100)
- ✅ Ideal answer generation for learning
- ✅ AI improvement suggestions
- ✅ PDF report download
- ✅ Industry templates (SWE, PM, Data)
- ✅ Multi-language support (10+ languages)

### Coming Next

- 📊 Session analytics & progress tracking
- 📈 Performance trends over time
- 🏆 Improvement leaderboard
- 💬 Follow-up Q&A with the AI coach
- 🎥 Live coaching dashboard
- 🔗 LinkedIn integration to share results

---

## 💻 Installation (Simplified)

**macOS/Linux:**

```bash
git clone <repo> && cd ai-interview-rehearsal-coach
python3 -m venv venv && source venv/bin/activate
pip install opencv-python-headless==4.11.0.80
pip install -r requirements.txt --break-system-packages
echo "GROQ_API_KEY=your_key" > .env
streamlit run app.py
```

**Get API Key:** https://console.groq.com/keys (free, no credit card)

---

## 📞 Support & Next Steps

**Want to Contribute?**

- Found a bug? Open an issue
- Have an idea? Start a discussion
- Want to add a feature? Pull requests welcome

**Learn More:**

- [Groq API Docs](https://console.groq.com/docs)
- [Streamlit Docs](https://docs.streamlit.io)

---

## 🎯 Bottom Line

This isn't just another tool. It's your personal interview coach that gives you:

1. **Honest feedback** on your interview skills (answer quality + delivery)
2. **Quantified progress** so you know you're improving
3. **Professional answers** to learn from
4. **Zero judgment** — practice as much as you need

**Start practicing today. Get hired tomorrow.**

---

**Made with ❤️ to help you ace your interviews**

---

## 👨‍💻 Author

**Anuj Patel**

AI × Productivity Enthusiast • Builder of calm, focused AI tools that help people learn, prepare, and grow.

🌐 **Connect & Follow:**

- 💼 [LinkedIn](https://www.linkedin.com/in/anujpatelofficial)
- 🐦 [Twitter / X](https://x.com/anuj_404)
- 📰 [Substack](https://buildprojectguide.substack.com)
- 🧠 [Hashnode](https://llmlearning.hashnode.dev)

---

**Version:** 1.0 POC  
**Last Updated:** October 30, 2025  
**Status:** Ready to Practice
