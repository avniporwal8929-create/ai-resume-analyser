# ATS Resume AI

A simple, fully functional, and beginner-friendly Applicant Tracking System (ATS) Resume Scorer and Advisor designed for college Machine Learning internships. 

This web application compares a candidate's resume (PDF, DOCX, TXT) against a Job Description to calculate compatibility, highlight matched vs. missing skills and keywords, and generate AI-driven improvement suggestions using the Google Gemini API.

---

## 🚀 Features

1. **Multi-format Resume Upload**: Supports `.pdf`, `.docx`, and `.txt` parsing using lightweight python libraries (`pdfplumber` and `python-docx`).
2. **Text Compatibility (TF-IDF)**: Computes a semantic similarity score using Scikit-Learn's `TfidfVectorizer` and `cosine_similarity`.
3. **Predefined Skill Matching**: Evaluates the resume against a list of technical skills loaded from `skills.json` and displays matched vs. missing skills.
4. **Keyword Matching**: Evaluates top-frequency terms from the Job Description against the resume text.
5. **Combined ATS Score**: Combines TF-IDF matching, skills matching, and keyword coverage into a single weighted score:
   $$\text{ATS Score} = 40\% \times \text{Cosine Match} + 40\% \times \text{Skill Match} + 20\% \times \text{Keyword Match}$$
6. **Grades**: Categorizes resumes into performance bands:
   - **90-100**: Excellent (Green)
   - **75-89**: Good (Blue)
   - **60-74**: Average (Yellow)
   - **Below 60**: Needs Improvement (Red)
7. **Matplotlib Visualizations**: Renders donut charts for overall scores, bar charts comparing metrics, and pie charts showing skill coverage.
8. **Generative AI Insights**: Integrates Google Gemini (`google-genai` SDK) to generate improvement suggestions, a custom professional summary, impact-driven bullet point rewrites (XYZ formula), interview tips, and long-term career advice.
9. **Exportable CSV Report**: Generates a downloadable CSV containing scores, matched/missing keywords, and Gemini AI suggestions.

---

## 📁 File Structure

```text
ATS-Resume-AI/
├── .streamlit/
│   └── config.toml          # Custom theme configuration (Blue theme)
├── assets/
│   └── .gitkeep             # Directory for local media/logos
├── ai_rewriter.py           # Google Gemini API connector & feedback parser
├── app.py                   # Main Streamlit web application dashboard
├── ats_scorer.py            # Similarity scoring, skills matching, and grading
├── README.md                # Project documentation and guide
├── requirements.txt         # List of Python dependencies
├── resume_parser.py         # File extraction logic for PDF, DOCX, and TXT
├── skills.json              # List of technical skills for analysis
└── utils.py                 # Core text processing, keyword and skill extractions
```

---

## 🛠️ Installation & Setup

Make sure you have Python 3.9+ installed on your computer.

### Step 1: Clone or copy this project to your local directory
Ensure all files are structured as shown in the file structure above.

### Step 2: Open a terminal inside the project root folder
Create a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
Install all the libraries listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 4: Configure Gemini API Key (Optional)
You can set your Gemini API key as an environment variable, or enter it directly in the application sidebar.
- **Windows Command Prompt**:
  ```cmd
  set GEMINI_API_KEY=your_actual_api_key_here
  ```
- **Windows PowerShell**:
  ```powershell
  $env:GEMINI_API_KEY="your_actual_api_key_here"
  ```
- **macOS / Linux**:
  ```bash
  export GEMINI_API_KEY="your_actual_api_key_here"
  ```

### Step 5: Run the Streamlit Application
Start the server and launch the web UI in your browser:
```bash
streamlit run app.py
```
If the browser does not open automatically, copy the Local URL (usually `http://localhost:8501`) from the terminal.

---

## 💻 Tech Stack Detail

- **Framework**: Streamlit (Web App GUI)
- **Natural Language Processing**: Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`)
- **Document Extractors**: `pdfplumber` (PDF text extract), `python-docx` (DOCX extraction)
- **Visuals**: Matplotlib (Custom plotting)
- **Generative AI**: Google GenAI SDK (`google-genai`) using the `gemini-2.0-flash` model.
