import requests
import re
import os
import time
import logging

# Configure module-level logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Currently supported, stable OpenRouter models (tried in order) ─────────────
# Priority: higher-quality models first, ultra-light free fallbacks last.
# Remove or add models here as OpenRouter availability changes.
MODELS_TO_TRY = [
    "google/gemini-2.0-flash-001",               # Stable Gemini 2.0 Flash
    "google/gemini-flash-1.5",                    # Gemini 1.5 Flash
    "meta-llama/llama-3.1-8b-instruct:free",      # Llama 3.1 8B (free tier)
    "mistralai/mistral-7b-instruct:free",          # Mistral 7B (free tier)
    "microsoft/phi-3-mini-128k-instruct:free",     # Phi-3 Mini (free tier)
]

# HTTP status codes that should trigger a retry on the next model
_RETRYABLE_STATUS_CODES = {404, 429, 500, 502, 503, 504}

# How many times to retry a single model before moving on
_MAX_RETRIES_PER_MODEL = 1

# Seconds to wait between retries
_RETRY_DELAY_SECONDS = 2

# Request timeout in seconds
_REQUEST_TIMEOUT = 60


class AIRewriter:
    """
    Wraps the OpenRouter chat-completion API.

    Reads the API key from Streamlit secrets (``OPENROUTER_API_KEY``) or the
    environment variable of the same name.  Call ``generate_feedback()`` or
    ``rewrite_text()`` to invoke AI features.
    """

    def __init__(self, api_key: str = None):
        """
        Initialises the AIRewriter.

        Parameters
        ----------
        api_key : str, optional
            Override the API key read from secrets / environment variables.
        """
        if api_key:
            self.api_key = api_key
        else:
            try:
                import streamlit as st
                self.api_key = (
                    st.secrets.get("OPENROUTER_API_KEY", "")
                    or os.getenv("OPENROUTER_API_KEY", "")
                )
            except Exception:
                self.api_key = os.getenv("OPENROUTER_API_KEY", "")

    # ── Private helpers ────────────────────────────────────────────────────────

    def _validate_api_key(self) -> None:
        """
        Raises a descriptive ``ValueError`` when no API key is configured,
        so callers can catch it and show a friendly Streamlit warning.
        """
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "Please configure your OpenRouter API key in Streamlit Secrets "
                "(key name: OPENROUTER_API_KEY) or as an environment variable."
            )

    def _call_openrouter(self, prompt: str) -> str:
        """
        Sends ``prompt`` to OpenRouter, cycling through ``MODELS_TO_TRY``.

        Behaviour
        ---------
        * Validates the API key before making any network call.
        * Retries each model ``_MAX_RETRIES_PER_MODEL`` times on transient errors.
        * Moves to the next model on 404 / 429 / 5xx responses.
        * Raises an ``Exception`` only after **all** models have been exhausted.
        """
        self._validate_api_key()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ai-resume-analyser.streamlit.app",
            "X-Title": "AI Resume Analyser",
        }

        last_error: str = "No models were attempted."

        for model in MODELS_TO_TRY:
            for attempt in range(1, _MAX_RETRIES_PER_MODEL + 2):  # +2 so range gives 1 & 2
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                }
                try:
                    logger.info("Trying model %s (attempt %d)…", model, attempt)
                    response = requests.post(
                        OPENROUTER_BASE_URL,
                        headers=headers,
                        json=payload,
                        timeout=_REQUEST_TIMEOUT,
                    )

                    # ── Success ───────────────────────────────────────────────
                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [])
                        if not choices:
                            last_error = f"Model {model} returned an empty 'choices' list."
                            logger.warning(last_error)
                            break  # try next model

                        content = choices[0].get("message", {}).get("content", "").strip()
                        if not content:
                            last_error = f"Model {model} returned an empty response body."
                            logger.warning(last_error)
                            break  # try next model

                        logger.info("Success with model %s.", model)
                        return content

                    # ── Invalid API key ───────────────────────────────────────
                    elif response.status_code == 401:
                        raise ValueError(
                            "Invalid OpenRouter API key (HTTP 401). "
                            "Please check your OPENROUTER_API_KEY in Streamlit Secrets."
                        )

                    # ── Retryable error — try next model ──────────────────────
                    elif response.status_code in _RETRYABLE_STATUS_CODES:
                        last_error = (
                            f"Model {model} returned HTTP {response.status_code}: "
                            f"{response.text[:200]}"
                        )
                        logger.warning(last_error)
                        if attempt <= _MAX_RETRIES_PER_MODEL:
                            logger.info("Waiting %ss before retry…", _RETRY_DELAY_SECONDS)
                            time.sleep(_RETRY_DELAY_SECONDS)
                            continue  # retry same model
                        else:
                            break  # move to next model

                    # ── Unexpected status — log and move on ───────────────────
                    else:
                        last_error = (
                            f"Model {model} returned unexpected HTTP "
                            f"{response.status_code}: {response.text[:200]}"
                        )
                        logger.warning(last_error)
                        break  # try next model

                # ── Network / timeout errors ──────────────────────────────────
                except requests.exceptions.Timeout:
                    last_error = f"Model {model} timed out after {_REQUEST_TIMEOUT}s."
                    logger.warning(last_error)
                    break  # try next model

                except requests.exceptions.ConnectionError as exc:
                    last_error = f"Network error with model {model}: {exc}"
                    logger.warning(last_error)
                    break  # try next model

                except ValueError:
                    # Re-raise auth errors immediately — no point trying other models
                    raise

                except Exception as exc:
                    last_error = f"Unexpected error with model {model}: {exc}"
                    logger.exception(last_error)
                    break  # try next model

        raise Exception(
            f"All models failed to return a valid response. Last error: {last_error}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate_feedback(self, resume_text: str, jd_text: str) -> str:
        """
        Analyses the resume against the job description and returns structured
        markdown feedback with five numbered sections.
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

    def rewrite_text(
        self, text_to_rewrite: str, jd_text: str = "", style: str = "XYZ Formula"
    ) -> str:
        """
        Rewrites a specific bullet point or section of a resume based on the
        job description and the selected style.
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


# ── Parsing helper ─────────────────────────────────────────────────────────────

def parse_gemini_feedback(feedback_text: str) -> dict:
    """
    Parses the AI markdown response into five distinct sections for display
    in Streamlit tabs or expanders.  Uses regex splitting on numbered headers.

    Returns
    -------
    dict
        Keys: ``suggestions``, ``summary``, ``bullets``, ``interview``, ``career``.
        If the response does not contain all five sections, the raw text is
        placed in ``suggestions`` and the remaining keys are empty strings.
    """
    sections = {
        "suggestions": "",
        "summary": "",
        "bullets": "",
        "interview": "",
        "career": "",
    }

    # Split by numbered markdown headers: # 1., # 2., ## 1., ## 2., etc.
    parts = re.split(r"#+\s+\d+\.\s+", feedback_text)

    # parts[0] is everything before the first numbered header (often empty/preamble)
    if len(parts) >= 6:
        sections["suggestions"] = parts[1].strip()
        sections["summary"] = parts[2].strip()
        sections["bullets"] = parts[3].strip()
        sections["interview"] = parts[4].strip()
        sections["career"] = parts[5].strip()
    else:
        # Fallback: show the whole response in the first tab
        sections["suggestions"] = feedback_text

    return sections
