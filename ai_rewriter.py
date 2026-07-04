from google import genai
import re

class AIRewriter:
    def __init__(self, api_key: str = None):
        """
        Initializes the GenAI Client.
        If api_key is provided, it configures it.
        Otherwise, client will search for GEMINI_API_KEY / GOOGLE_API_KEY environment variables.
        """
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

    def generate_feedback(self, resume_text: str, jd_text: str) -> str:
        """
        Sends the Resume and Job Description to the Gemini-2.0-Flash model.
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

        try:
            # Using gemini-2.0-flash as the fast, efficient recommended model
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            raise Exception(f"Failed to generate feedback from Gemini API: {str(e)}")

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
        try:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            raise Exception(f"Failed to rewrite text using Gemini API: {str(e)}")


def parse_gemini_feedback(feedback_text: str) -> dict:
    """
    Parses the Markdown response from Gemini into five distinct parts
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
