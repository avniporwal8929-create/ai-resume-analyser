import re

# Standard English stopwords to avoid external dependencies
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'arent', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'cant', 'cannot', 'could',
    'couldnt', 'did', 'didnt', 'do', 'does', 'doesnt', 'doing', 'dont', 'down', 'during', 'each', 'few', 'for',
    'from', 'further', 'had', 'hadnt', 'has', 'hasnt', 'have', 'havent', 'having', 'he', 'hed', 'hell', 'hes',
    'her', 'here', 'heres', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'ill', 'im',
    'ive', 'if', 'in', 'into', 'is', 'isnt', 'it', 'its', 'itself', 'lets', 'me', 'more', 'most', 'mustnt', 'my',
    'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'same', 'shant', 'she', 'shed', 'shell', 'shes', 'should', 'shouldnt',
    'so', 'some', 'such', 'than', 'that', 'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then',
    'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasnt', 'we', 'wed', 'well', 'were', 'weve', 'werent',
    'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who', 'whos', 'whom', 'why', 'whys',
    'with', 'wont', 'would', 'wouldnt', 'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself',
    'yourselves'
}

def clean_text(text: str) -> str:
    """
    Cleans the input text by:
    1. Converting to lowercase.
    2. Replacing newlines and tabs with spaces.
    3. Removing special characters but preserving programming terms like C++, C#, .NET.
    4. Collapsing multiple spaces.
    """
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Replace newlines and carriage returns with spaces
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Remove most punctuation but keep chars relevant to tech skills
    # Keeps letters, numbers, spaces, dots, pluses, and hashes.
    text = re.sub(r'[^a-zA-Z0-9\s\.\+#\-]', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize(text: str) -> list[str]:
    """
    Cleans and tokenizes text into words.
    """
    cleaned = clean_text(text)
    return cleaned.split()

def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Removes standard stopwords from a list of tokens.
    """
    return [t for t in tokens if t not in STOPWORDS]

def extract_keywords(text: str, top_n: int = 20) -> list[str]:
    """
    Extracts top N keywords based on term frequency from cleaned text,
    excluding common stopwords and numeric/short words.
    """
    tokens = tokenize(text)
    filtered = remove_stopwords(tokens)
    
    # Filter out single/double letter words and pure numbers
    filtered = [t for t in filtered if len(t) > 2 and not t.isdigit()]
    
    # Calculate word frequency
    freq_map = {}
    for word in filtered:
        freq_map[word] = freq_map.get(word, 0) + 1
        
    # Sort descending by frequency
    sorted_words = sorted(freq_map.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:top_n]]

def extract_skills(text: str, skills_list: list[str]) -> list[str]:
    """
    Scans the text for technical skills from skills_list.
    Uses positive/negative lookarounds to enforce word boundaries even for skills
    containing special characters (like C++, C#, .NET).
    """
    text_lower = text.lower()
    found_skills = []
    
    for skill in skills_list:
        skill_lower = skill.lower()
        skill_escaped = re.escape(skill_lower)
        
        # Build boundary condition: must not be preceded or followed by alphanumeric chars.
        # This protects 'Go' in 'Google' and 'Java' in 'Javascript'.
        pattern = r'(?<![a-zA-Z0-9])' + skill_escaped + r'(?![a-zA-Z0-9])'
        
        if re.search(pattern, text_lower):
            if skill not in found_skills:
                found_skills.append(skill)
                
    return found_skills
