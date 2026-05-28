# gemini_ai.py
# Uses Google Gemini to generate treatment advice for detected diseases

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Setup Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")  # free tier model

def get_treatment(disease_name: str, confidence: float) -> str:
    """
    Ask Gemini AI for treatment advice based on detected disease
    Returns: treatment text string
    """
    if "healthy" in disease_name.lower():
        return "✅ Your plant appears healthy! Keep up with regular watering, proper sunlight, and periodic fertilization."
    
    prompt = f"""
    A crop disease detection system identified the following plant disease:
    
    Disease: {disease_name}
    Detection Confidence: {confidence}%
    
    Please provide:
    1. Brief description of this disease (2-3 sentences)
    2. Immediate treatment steps (3-4 bullet points)
    3. Preventive measures for the future (2-3 bullet points)
    4. Whether organic or chemical treatment is recommended
    
    Keep the response concise, practical, and farmer-friendly.
    Format with clear headings and bullet points.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return f"""
        **Treatment for {disease_name}:**
        
        Please consult your local agricultural extension officer for treatment advice.
        
        General steps:
        • Remove and destroy infected plant parts
        • Improve air circulation around plants
        • Avoid overhead watering
        • Apply appropriate fungicide/pesticide as recommended locally
        """