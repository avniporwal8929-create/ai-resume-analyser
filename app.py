import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import io
import csv
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file (for local development)
load_dotenv()


# Import project modules
from utils import clean_text, extract_skills, extract_keywords
from resume_parser import parse_resume
from ats_scorer import ATSScorer
from ai_rewriter import AIRewriter, parse_gemini_feedback

# Page configuration
st.set_page_config(
    page_title="ATS Resume AI - Resume Scorer & Advisor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look & feel
st.markdown("""
<style>
    /* Main body background color */
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Custom Card Style */
    .metric-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-title {
        margin: 0;
        font-size: 0.85rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.075em;
    }
    .metric-value {
        margin: 10px 0 0 0;
        font-size: 2.25rem;
        font-weight: 800;
    }
    
    /* HTML Badges for Skills and Keywords */
    .badge-green {
        background-color: #dcfce7;
        color: #15803d;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        margin: 4px;
        display: inline-block;
        border: 1px solid #bbf7d0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .badge-red {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.875rem;
        margin: 4px;
        display: inline-block;
        border: 1px solid #fecaca;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Card details container */
    .section-card {
        background-color: #ffffff;
        padding: 28px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        margin-bottom: 20px;
    }
    
    /* Header title design */
    .header-title {
        color: #1e3a8a;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 5px;
    }
    .header-subtitle {
        color: #475569;
        font-size: 1.12rem;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Helper to load skills database
@st.cache_data
def load_skills():
    skills_file = "skills.json"
    if os.path.exists(skills_file):
        try:
            with open(skills_file, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading skills.json: {str(e)}")
            
    # Fallback default skills list if file is missing or corrupt
    return [
        "Python", "Java", "C++", "C#", "C", "JavaScript", "TypeScript", "SQL",
        "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Django", "Flask",
        "FastAPI", "Machine Learning", "Deep Learning", "Data Analysis",
        "Data Science", "Scikit-Learn", "TensorFlow", "PyTorch", "Pandas",
        "NumPy", "Matplotlib", "Tableau", "Git", "Docker", "Kubernetes", "AWS",
        "Azure", "GCP", "DevOps", "CI/CD", "Linux", "MySQL", "PostgreSQL",
        "MongoDB", "REST", "Agile", "Scrum"
    ]

# Helper to generate CSV
def generate_csv_report(scores_dict, matched_skills, missing_skills, ai_feedback):
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["ATS Resume AI - Analysis Report"])
    writer.writerow(["Generated On", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    
    writer.writerow(["1. METRICS & OVERALL SCORES"])
    writer.writerow(["Overall ATS Score", f"{scores_dict['ats_score']}%"])
    writer.writerow(["Word Overlap Score", f"{scores_dict['overlap_score']}%"])
    writer.writerow(["Skill Match Score", f"{scores_dict['skill_score']}%"])
    writer.writerow(["Keyword Match Score", f"{scores_dict['keyword_score']}%"])
    writer.writerow(["Resume Grade", scores_dict['grade']])
    writer.writerow([])
    
    writer.writerow(["2. SKILLS ANALYSIS"])
    writer.writerow(["Matched Skills Count", len(matched_skills)])
    writer.writerow(["Matched Skills", ", ".join(matched_skills) if matched_skills else "None"])
    writer.writerow(["Missing Skills Count", len(missing_skills)])
    writer.writerow(["Missing Skills", ", ".join(missing_skills) if missing_skills else "None"])
    writer.writerow([])
    
    writer.writerow(["3. KEYWORDS ANALYSIS"])
    writer.writerow(["Matched Keywords Count", len(scores_dict['matched_keywords'])])
    writer.writerow(["Matched Keywords", ", ".join(scores_dict['matched_keywords']) if scores_dict['matched_keywords'] else "None"])
    writer.writerow(["Missing Keywords Count", len(scores_dict['missing_keywords'])])
    writer.writerow(["Missing Keywords", ", ".join(scores_dict['missing_keywords']) if scores_dict['missing_keywords'] else "None"])
    writer.writerow([])
    
    writer.writerow(["4. AI SUGGESTIONS & FEEDBACK"])
    if ai_feedback:
        clean_feedback = ai_feedback.replace("\r\n", "\n")
        writer.writerow(["AI Recommendations", clean_feedback])
    else:
        writer.writerow(["AI Recommendations", "Not generated (API key not configured)"])
        
    return output.getvalue()

# Helper colors for score grading
def get_score_color(score):
    if score >= 90:
        return '#10b981' # Green (Emerald)
    elif score >= 75:
        return '#3b82f6' # Blue (Blue-500)
    elif score >= 60:
        return '#f59e0b' # Orange (Amber-500)
    else:
        return '#ef4444' # Red (Red-500)

# Load skills list
skills_list = load_skills()

# SIDEBAR IMPLEMENTATION
st.sidebar.markdown("<h2 style='color:#1e3a8a;font-weight:700;'>Configuration</h2>", unsafe_allow_html=True)
st.sidebar.info("Upload your resume and paste the job description to test compatibility.")

# File Uploader
st.sidebar.markdown("### 1. Upload Resume")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file", 
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, TXT. Scanned PDFs are not supported."
)

# Gemini API key is provided via environment variable `GEMINI_API_KEY` or a hard‑coded fallback.
# No UI input is required.
# The variable `gemini_key` is no longer used; the app will read `os.getenv('GEMINI_API_KEY')` directly.

# Sidebar footer / Clear Button
st.sidebar.markdown("---")
if st.sidebar.button("Clear All Inputs", use_container_width=True):
    st.rerun()

# MAIN AREA IMPLEMENTATION

# Header Section
st.markdown("<div class='header-title'>ATS Resume AI Optimizer</div>", unsafe_allow_html=True)
st.markdown("<div class='header-subtitle'>Upload your resume, analyze alignment with job requirements, and get professional suggestions to boost interview selection chances.</div>", unsafe_allow_html=True)

# Layout for inputs: 2 Column Layout (Left: Instructions/Info, Right: Job Description input)
input_col1, input_col2 = st.columns([1, 1.2])

with input_col1:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("### 🛠️ How to Use:")
    st.markdown("""
    1. **Upload Resume**: In the sidebar, select your Resume in `.pdf`, `.docx`, or `.txt` format.
    2. **Paste Job Description**: Paste the target Job Description in the text area on the right.
    3. **Set API Key (Optional)**: Provide a Gemini API key in the sidebar for AI improvement reviews and interview prep.
    4. **Analyze**: Click the **Analyze Resume** button to run calculations.
    5. **Review Results**: Use the interactive tabs to explore matching keywords, charts, and downloadable reports.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with input_col2:
    st.markdown("<div class='section-card' style='padding-top:15px; padding-bottom:15px;'>", unsafe_allow_html=True)
    jd_text = st.text_area(
        "📋 Paste Job Description", 
        height=250, 
        placeholder="Paste the requirements or description of the job here...",
        help="Paste the full job requirements. Word frequencies, skills, and semantics will be extracted from this text."
    )
    # Character Counter
    char_count = len(jd_text)
    st.markdown(f"<p style='text-align: right; color:#64748b; font-size:0.8rem; margin:0;'>Character count: {char_count}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Analyze Button
st.markdown("<div style='text-align: center; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)
run_analysis = st.button("🚀 Analyze Resume", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if run_analysis:
    # Error Handling for missing inputs
    if not uploaded_file:
        st.error("❌ Please upload a resume file (PDF, DOCX, or TXT) in the sidebar first.")
    elif not jd_text.strip():
        st.error("❌ Please paste a Job Description in the text field first.")
    else:
        with st.spinner("Extracting text and calculating compatibility scores..."):
            # 1. Parse Resume Text
            try:
                resume_text = parse_resume(uploaded_file)
            except Exception as e:
                st.error(f"❌ Resume parsing failed: {str(e)}")
                st.stop()

            # 2. Run ATS Scorer
            scorer = ATSScorer(skills_list)
            scores = scorer.calculate_scores(resume_text, jd_text)
            
            # Save results to session state so they persist if the user changes tabs
            st.session_state['analysis_done'] = True
            st.session_state['scores'] = scores
            st.session_state['resume_text'] = resume_text
            st.session_state['jd_text'] = jd_text
            # 3. Call AI via OpenRouter (key is built into AIRewriter)
            st.session_state['ai_feedback'] = None
            st.session_state['ai_success'] = False

            with st.spinner("✨ Generating AI-powered resume suggestions..."):
                try:
                    rewriter = AIRewriter()
                    raw_feedback = rewriter.generate_feedback(resume_text, jd_text)
                    st.session_state['ai_feedback'] = raw_feedback
                    st.session_state['ai_success'] = True
                except Exception as e:
                    st.warning(f"⚠️ AI call failed: {str(e)}. You can still inspect calculations and keyword counts.")


# Check if analysis has run and data is saved in session state
if st.session_state.get('analysis_done', False):
    scores = st.session_state['scores']
    resume_text = st.session_state['resume_text']
    jd_text = st.session_state['jd_text']
    ai_feedback = st.session_state.get('ai_feedback', None)
    ai_success = st.session_state.get('ai_success', False)
    
    # Define color matching for the grade
    score_color = get_score_color(scores['ats_score'])
    
    # SUCCESS BANNER
    st.success("🎉 Analysis completed successfully! Review the breakdown below.")
    
    # 4 INTERACTIVE TABS
    tab_dashboard, tab_keywords, tab_ai, tab_export = st.tabs([
        "📊 Dashboard", 
        "🔍 Keywords & Skills", 
        "🤖 AI Suggestions", 
        "📥 Report & Export"
    ])
    
    # --- TAB 1: DASHBOARD ---
    with tab_dashboard:
        # Side-by-side metric cards
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-title'>Overall ATS Score</p>
                <p class='metric-value' style='color:{score_color};'>{scores['ats_score']}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class='metric-card'>
                 <p class='metric-title'>Word Overlap</p>
                 <p class='metric-value' style='color:#3b82f6;'>{scores['overlap_score']}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <p class='metric-title'>Resume Grade</p>
                <p class='metric-value' style='color:{score_color};'>{scores['grade']}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Divider
        st.markdown("---")
        
        # Visualizations (Matplotlib Charts)
        chart_col1, chart_col2 = st.columns([1, 1.2])
        
        with chart_col1:
            st.markdown("<h4 style='text-align: center; color:#1e3a8a;'>Overall Score Breakdown</h4>", unsafe_allow_html=True)
            # Create Donut Chart
            fig_donut, ax_donut = plt.subplots(figsize=(4, 4))
            sizes = [scores['ats_score'], 100 - scores['ats_score']]
            colors = [score_color, '#e2e8f0']
            
            # Donut chart
            wedges, texts = ax_donut.pie(
                sizes, 
                colors=colors, 
                startangle=90, 
                counterclock=False, 
                wedgeprops=dict(width=0.28, edgecolor='white')
            )
            ax_donut.text(
                0, 0, f"{scores['ats_score']}%\nScore", 
                ha='center', va='center', 
                fontsize=18, fontweight='bold', 
                color=score_color
            )
            ax_donut.axis('equal')
            fig_donut.patch.set_facecolor('#ffffff')
            st.pyplot(fig_donut)
            plt.close(fig_donut)
            
        with chart_col2:
            st.markdown("<h4 style='text-align: center; color:#1e3a8a;'>Score Categories comparison</h4>", unsafe_allow_html=True)
            # Create Bar Chart
            fig_bar, ax_bar = plt.subplots(figsize=(6, 3.5))
            categories = ['Word Overlap', 'Skill Match', 'Keyword Match', 'Weighted ATS']
            values = [scores['overlap_score'], scores['skill_score'], scores['keyword_score'], scores['ats_score']]
            bar_colors = ['#60a5fa', '#c084fc', '#fbbf24', score_color]
            
            bars = ax_bar.barh(categories, values, color=bar_colors, height=0.5)
            
            # Design details
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['bottom'].set_visible(False)
            ax_bar.spines['left'].set_color('#cbd5e1')
            ax_bar.grid(axis='x', linestyle='--', alpha=0.5)
            
            # Add labels to the right of bars
            for bar in bars:
                width = bar.get_width()
                ax_bar.text(
                    width + 2, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', 
                    ha='left', va='center', 
                    fontsize=9, fontweight='bold',
                    color='#334155'
                )
            
            ax_bar.set_xlim(0, 110)
            fig_bar.patch.set_facecolor('#ffffff')
            st.pyplot(fig_bar)
            plt.close(fig_bar)
            
    # --- TAB 2: KEYWORDS & SKILLS ---
    with tab_keywords:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Skills and Keywords Analysis")
        st.write("Below are the technical skills and core keywords found in the Job Description, matched against your resume.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Skill Overview and Pie Chart
        k_col1, k_col2 = st.columns([1, 1.2])
        
        with k_col1:
            st.markdown("#### Skills Summary")
            total_jd_skills = len(scores['matched_skills']) + len(scores['missing_skills'])
            
            st.write(f"💼 **Total Skills identified in Job Description**: {total_jd_skills}")
            st.write(f"✅ **Matched Skills**: {len(scores['matched_skills'])}")
            st.write(f"❌ **Missing Skills**: {len(scores['missing_skills'])}")
            
            # Matplotlib Pie Chart
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
            if total_jd_skills > 0:
                labels = ['Matched', 'Missing']
                sizes = [len(scores['matched_skills']), len(scores['missing_skills'])]
                pie_colors = ['#10b981', '#ef4444']
                explode = (0.05, 0) if len(scores['matched_skills']) > 0 and len(scores['missing_skills']) > 0 else (0, 0)
                
                ax_pie.pie(
                    sizes, 
                    explode=explode, 
                    labels=labels, 
                    colors=pie_colors, 
                    autopct='%1.1f%%',
                    startangle=140,
                    textprops={'fontsize': 10, 'weight': 'bold'}
                )
                ax_pie.axis('equal')
            else:
                ax_pie.text(0.5, 0.5, "No technical skills found\nin Job Description", ha='center', va='center')
                ax_pie.axis('off')
                
            fig_pie.patch.set_facecolor('#ffffff')
            st.pyplot(fig_pie)
            plt.close(fig_pie)
            
        with k_col2:
            st.markdown("#### Technical Skills breakdown")
            
            st.markdown("**✅ Matched Skills**")
            if scores['matched_skills']:
                skills_html = "".join([f"<span class='badge-green'>{skill}</span>" for skill in scores['matched_skills']])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.info("No matching skills found. Highlight relevant tools in your resume.")
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("**❌ Missing Skills**")
            if scores['missing_skills']:
                missing_skills_html = "".join([f"<span class='badge-red'>{skill}</span>" for skill in scores['missing_skills']])
                st.markdown(missing_skills_html, unsafe_allow_html=True)
            else:
                st.success("Excellent! You match all identified technical skills in the Job Description.")
                
        st.markdown("---")
        
        # Keyword Analysis Section
        st.markdown("#### 🔑 Keyword Analysis (Vocabulary Match)")
        st.write("These represent the most frequent terms extracted from the Job Description. Adding missing terms to your resume increases similarity score.")
        
        kw_col1, kw_col2 = st.columns(2)
        
        with kw_col1:
            st.markdown("**✅ Matched Keywords**")
            if scores['matched_keywords']:
                matched_kw_html = "".join([f"<span class='badge-green'>{kw}</span>" for kw in scores['matched_keywords']])
                st.markdown(matched_kw_html, unsafe_allow_html=True)
            else:
                st.info("No frequent JD keywords matched.")
                
        with kw_col2:
            st.markdown("**❌ Missing Keywords**")
            if scores['missing_keywords']:
                missing_kw_html = "".join([f"<span class='badge-red'>{kw}</span>" for kw in scores['missing_keywords']])
                st.markdown(missing_kw_html, unsafe_allow_html=True)
            else:
                st.success("Awesome! You matched all major words in the Job Description.")
                
    # --- TAB 3: AI SUGGESTIONS ---
    with tab_ai:
        if ai_success and ai_feedback:
            # Parse response
            parsed = parse_gemini_feedback(ai_feedback)
            
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("### 🤖 Generative AI Insights")
            st.write("Gemini has parsed your resume and generated recommendations targeted specifically for this Job Description.")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show sections in expandable items
            with st.expander("💡 1. Resume Improvement Suggestions", expanded=True):
                st.markdown(parsed["suggestions"])
                
            with st.expander("📝 2. Tailored Professional Summary", expanded=False):
                st.markdown(parsed["summary"])
                
            with st.expander("🎯 3. Better Resume Bullet Points (XYZ Formula)", expanded=False):
                st.markdown(parsed["bullets"])
                
            with st.expander("❓ 4. Interview Preparation Tips", expanded=False):
                st.markdown(parsed["interview"])
                
            with st.expander("🚀 5. Career Advice", expanded=False):
                st.markdown(parsed["career"])
        else:
            st.warning("⚠️ AI suggestions could not be generated. Please try running the analysis again.")
            
    # --- TAB 4: EXPORT REPORT ---
    with tab_export:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("### 📥 Download Analysis Report")
        st.write("Download a detailed CSV report summarizing the compatibility score, matched/missing skills, keyword analysis, and AI suggestions.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # CSV String Generator
        csv_data = generate_csv_report(
            scores, 
            scores['matched_skills'], 
            scores['missing_skills'], 
            ai_feedback
        )
        
        # Download Button
        st.download_button(
            label="Download Complete CSV Report",
            data=csv_data,
            file_name=f"ATS_Resume_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Preview Table
        st.markdown("#### 📄 Report Preview")
        preview_data = {
            "Metric Name": [
                "Overall ATS Score",
                "Text Compatibility (TF-IDF)",
                "Skills Match Score",
                "Keywords Match Score",
                "Resume Grade",
                "Matched Skills Count",
                "Missing Skills Count",
                "Matched Keywords Count",
                "Missing Keywords Count"
            ],
            "Value / Count": [
                f"{scores['ats_score']}%",
                f"{scores['overlap_score']}%",
                f"{scores['skill_score']}%",
                f"{scores['keyword_score']}%",
                scores['grade'],
                len(scores['matched_skills']),
                len(scores['missing_skills']),
                len(scores['matched_keywords']),
                len(scores['missing_keywords'])
            ]
        }
        df_preview = pd.DataFrame(preview_data)
        st.table(df_preview)
