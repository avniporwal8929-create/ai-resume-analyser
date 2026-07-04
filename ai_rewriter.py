import requests
import re
import os

def _get_api_key():
    """Read OpenRouter key from Streamlit secrets (cloud) or .env (local)."""
    try:
        import streamlit as st
        return st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))
    except Exception:
        return os.getenv("OPENROUTER_API_KEY", "")

OPENROUTER_API_KEY = _get_api_key()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free models available on OpenRouter - tried in order
MODELS_TO_TRY = [
    "google/gemini-2.0-flash-exp:free",
    "google/gemini-flash-1.5-8b",
    "meta-llama/llama-3.1-8b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
]


class AIRewriter:
    def __init__(self, api_key: str = None):
        """
        Initializes the AIRewriter using OpenRouter API.
        api_key parameter kept for backward compatibility but OpenRouter key is used by default.
        """
        self.api_key = OPENROUTER_API_KEY

    def _call_openrouter(self, prompt: str) -> str:
        """
        Sends a prompt to OpenRouter, trying multiple free models until one succeeds.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-resume-analyser.streamlit.app",
            "X-Title": "AI Resume Analyser"
        }

        last_error = None
        for model in MODELS_TO_TRY:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            }
            try:
                response = requests.post(
                    OPENROUTER_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                if response.status_code == 200:
                    data = response.json()
                    return data["choices"][0]["message"]["content"]
                elif response.status_code in (429, 503):
                    # Quota/rate-limit — try next model
                    last_error = f"Model {model} returned {response.status_code}"
                    continue
                else:
                    last_error = f"Model {model} returned {response.status_code}: {response.text}"
                    break
            except Exception as e:
                last_error = str(e)
                continue

        raise Exception(f"All models failed. Last error: {last_error}")

    def generate_feedback(self, resume_text: str, jd_text: str) -> str:
        """
        Sends the Resume and Job Description to an AI model via OpenRouter.
        Prompts the model to return 5 distinct, numbered headers in markdown format.
        """
        prompt = f"""
        You are an expert tech recruiter and resume optimizer. 
        Analyze the candidate's resume and compare it against the Job Description (JD) provided below. 
        Provide constructive, direct, and actionable feedback to help the candidate improve their application.

        Crucial Instructions:
        - Do NOT rewrite the entire resume.
        - Give clear, structured, and bulleted recommendations.
        - Use the XYZ formula for bullet point improvements: "Accomplished [X], as measured by [Y], by doing [Z]".

        Please format your output strictly in markdown, with these five main headings:

        # 1. Resume Improvement Suggestions
        [List 3-5 specific suggestions on how to improve structure, style, or content matching the JD.]

        # 2. Professional Summary
        [Write a compelling, 3-4 sentence professional summary tailored to this Job Description that the candidate can use at the top of their resume.]

        # 3. Better Resume Bullet Points
        [Identify 2-3 weak or generic bullet points standard in resumes, and show how they can be rewritten to be more impact-driven, using the XYZ formula.]

        # 4. Interview Preparation Tips
        [Provide 3-4 tailored interview questions the candidate is likely to face for this role, along with guidance on how they should answer based on their resume.]

        # 5. Career Advice
        [Give strategic, long-term career advice for someone looking to grow into this type of role (e.g., certifications, projects to build, skills to learn next).]

        ---
        RESUME:
        {resume_text}

        ---
        JOB DESCRIPTION:
        {jd_text}
        """
        return self._call_openrouter(prompt)

    def rewrite_text(self, text_to_rewrite: str, jd_text: str = "", style: str = "XYZ Formula") -> str:
        """
        Rewrites a specific bullet point or section of a resume based on the JD and selected style.
        """
        prompt = f"""
        You are an expert resume writer and career coach.
        Rewrite the following text from a resume to make it more professional, impact-driven, and aligned with the target job requirements.
        
        Text to rewrite:
        "{text_to_rewrite}"
        
        Target Job Description / Requirements (if provided):
        "{jd_text}"
        
        Style / Focus:
        {style}
        
        Instructions:
        - If the style is "XYZ Formula", follow Google's XYZ formula: "Accomplished [X], as measured by [Y], by doing [Z]".
        - Keep the rewritten version concise (typically 1-2 sentences or bullet points).
        - Maintain truthfulness but maximize professional impact.
        - Provide 2-3 variations of the rewritten text.
        - Do NOT include any introductory or concluding conversational text. Output only the rewritten variations formatted in markdown as a list.
        """
        return self._call_openrouter(prompt)


def parse_gemini_feedback(feedback_text: str) -> dict:
    """
    Parses the Markdown response into five distinct parts
    for display in Streamlit tabs or expanders.
    Uses regex splitting based on markdown headers.
    """
    sections = {
        "suggestions": "",
        "summary": "",
        "bullets": "",
        "interview": "",
        "career": ""
    }

    # Split by standard headers: # 1., # 2., etc. or ## 1., ## 2.
    parts = re.split(r'#+\s+\d+\.\s+', feedback_text)
    
    # parts[0] is everything before "# 1. " (often empty or preamble)
    if len(parts) >= 6:
        sections["suggestions"] = parts[1].strip()
        sections["summary"] = parts[2].strip()
        sections["bullets"] = parts[3].strip()
        sections["interview"] = parts[4].strip()
        sections["career"] = parts[5].strip()
    else:
        # If parsing fails because model output structure differed,
        # put the whole response in the first tab and leave others empty
        sections["suggestions"] = feedback_text
        
    return sections
