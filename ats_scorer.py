import re
from utils import clean_text, extract_skills, extract_keywords, tokenize, remove_stopwords

class ATSScorer:
    def __init__(self, skills_list: list[str]):
        """
        Initializes the scorer with a predefined skills database list.
        """
        self.skills_list = skills_list

    def calculate_scores(self, resume_text: str, jd_text: str) -> dict:
        """
        Main interface to calculate all ATS-related metrics:
        - Word Overlap Score
        - Skill Overlap (Matched, Missing, Total, Skill Score)
        - Keyword Overlap (Matched, Missing, Keyword Score)
        - Combined Weighted ATS Score (0-100)
        - Performance Grade
        """
        # Clean both texts
        cleaned_resume = clean_text(resume_text)
        cleaned_jd = clean_text(jd_text)

        # Handle empty fields edge cases
        if not cleaned_resume or not cleaned_jd:
            return {
                "overlap_score": 0.0,
                "skill_score": 0.0,
                "keyword_score": 0.0,
                "ats_score": 0.0,
                "matched_skills": [],
                "missing_skills": [],
                "matched_keywords": [],
                "missing_keywords": [],
                "grade": "Needs Improvement"
            }

        # 1. Word Overlap Match Score
        # Tokenize and remove stopwords to extract unique meaningful words
        resume_tokens = set(remove_stopwords(tokenize(cleaned_resume)))
        jd_tokens = set(remove_stopwords(tokenize(cleaned_jd)))

        if jd_tokens:
            overlap_words = jd_tokens.intersection(resume_tokens)
            overlap_score = round((len(overlap_words) / len(jd_tokens)) * 100, 2)
        else:
            overlap_score = 100.0

        # 2. Skill Match Analysis
        # Extract skills present in JD
        jd_skills = extract_skills(jd_text, self.skills_list)
        # Extract skills present in Resume
        resume_skills = extract_skills(resume_text, self.skills_list)

        jd_skills_set = set(jd_skills)
        resume_skills_set = set(resume_skills)

        # Overlaps
        matched_skills = sorted(list(jd_skills_set.intersection(resume_skills_set)))
        missing_skills = sorted(list(jd_skills_set.difference(resume_skills_set)))

        # Skill Score calculation
        if jd_skills_set:
            skill_score = round((len(matched_skills) / len(jd_skills_set)) * 100, 2)
        else:
            # If JD lists no skills from our list, default to 100% skill score
            skill_score = 100.0

        # 3. Keyword Match Analysis
        # Get top 20 keywords from the JD
        jd_keywords = extract_keywords(jd_text, top_n=20)
        
        matched_keywords = []
        missing_keywords = []

        for kw in jd_keywords:
            # Search for keyword in the cleaned resume with word boundaries
            # Using custom lookaround to handle special characters in keywords
            kw_escaped = re.escape(kw.lower())
            pattern = r'(?<![a-zA-Z0-9])' + kw_escaped + r'(?![a-zA-Z0-9])'
            
            if re.search(pattern, cleaned_resume):
                matched_keywords.append(kw)
            else:
                missing_keywords.append(kw)

        # Keyword Score calculation
        if jd_keywords:
            keyword_score = round((len(matched_keywords) / len(jd_keywords)) * 100, 2)
        else:
            keyword_score = 100.0

        # 4. Final Weighted ATS Score (0 - 100)
        # Formula: 40% Word Overlap Match, 40% Skill Match, 20% Keyword Match
        ats_score = round((0.4 * overlap_score) + (0.4 * skill_score) + (0.2 * keyword_score), 2)

        # Ensure bounds
        ats_score = min(max(ats_score, 0.0), 100.0)

        # 5. Grade Assignment
        # 90-100 = Excellent
        # 75-89 = Good
        # 60-74 = Average
        # Below 60 = Needs Improvement
        if ats_score >= 90:
            grade = "Excellent"
        elif ats_score >= 75:
            grade = "Good"
        elif ats_score >= 60:
            grade = "Average"
        else:
            grade = "Needs Improvement"

        return {
            "overlap_score": overlap_score,
            "skill_score": skill_score,
            "keyword_score": keyword_score,
            "ats_score": ats_score,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "matched_keywords": matched_keywords,
            "missing_keywords": missing_keywords,
            "grade": grade
        }
