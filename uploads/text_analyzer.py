"""
Text Sentiment Analysis Module
Analyzes text input to detect stress levels based on sentiment
"""
from textblob import TextBlob
import re

from config import TEXT_STRESS_THRESHOLDS


# Stress-related keywords with weights
STRESS_KEYWORDS = {
    'high_stress': [
        'overwhelmed', 'anxious', 'depressed', 'exhausted', 'hopeless',
        'terrible', 'awful', 'stressed', 'panic', 'worried', 'scared',
        'frustrated', 'angry', 'helpless', 'desperate', 'miserable',
        'burned out', 'burnout', 'can\'t cope', 'breaking down', 'crying',
        'insomnia', 'nightmare', 'suicidal', 'worthless', 'failure'
    ],
    'moderate_stress': [
        'tired', 'busy', 'pressure', 'deadline', 'difficult', 'hard',
        'challenging', 'concern', 'nervous', 'uneasy', 'tense',
        'uncomfortable', 'struggling', 'tough', 'demanding', 'hectic'
    ],
    'low_stress': [
        'happy', 'relaxed', 'calm', 'peaceful', 'content', 'good',
        'great', 'wonderful', 'excellent', 'fantastic', 'amazing',
        'grateful', 'blessed', 'optimistic', 'confident', 'energetic',
        'motivated', 'excited', 'joyful', 'comfortable', 'balanced'
    ]
}


class TextStressAnalyzer:
    """
    Analyzes text to determine stress levels using sentiment analysis
    and keyword detection
    """
    
    def __init__(self):
        self.thresholds = TEXT_STRESS_THRESHOLDS
    
    def preprocess_text(self, text):
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_sentiment(self, text):
        """
        Get sentiment scores using TextBlob
        
        Returns:
            tuple: (polarity, subjectivity)
            - polarity: -1 (negative) to 1 (positive)
            - subjectivity: 0 (objective) to 1 (subjective)
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    def detect_keywords(self, text):
        """
        Detect stress-related keywords in text
        
        Returns:
            dict with keyword counts for each stress level
        """
        text_lower = text.lower()
        
        results = {
            'high_stress': [],
            'moderate_stress': [],
            'low_stress': []
        }
        
        for level, keywords in STRESS_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    results[level].append(keyword)
        
        return results
    
    def calculate_keyword_score(self, keywords_found):
        """
        Calculate stress score based on keywords found
        
        Returns:
            float: score from -1 (low stress) to 1 (high stress)
        """
        high_count = len(keywords_found['high_stress'])
        moderate_count = len(keywords_found['moderate_stress'])
        low_count = len(keywords_found['low_stress'])
        
        total = high_count + moderate_count + low_count
        
        if total == 0:
            return 0  # Neutral
        
        # Weighted score
        score = (high_count * 1.0 + moderate_count * 0.3 - low_count * 0.5) / total
        return max(-1, min(1, score))  # Clamp to [-1, 1]
    
    def analyze(self, text):
        """
        Full analysis pipeline
        
        Args:
            text: Input text string
        
        Returns:
            dict with analysis results
        """
        if not text or len(text.strip()) < 3:
            return {
                'success': False,
                'error': 'Please enter some text to analyze'
            }
        
        # Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Get sentiment
        polarity, subjectivity = self.get_sentiment(cleaned_text)
        
        # Detect keywords
        keywords_found = self.detect_keywords(cleaned_text)
        keyword_score = self.calculate_keyword_score(keywords_found)
        
        # Combine scores (70% sentiment, 30% keywords)
        combined_score = (polarity * 0.7) + (-keyword_score * 0.3)
        
        # Determine stress level
        if combined_score < self.thresholds['high']:
            stress_level = 3
            stress_text = "High Stress"
            stress_score = 8
        elif combined_score < self.thresholds['moderate']:
            stress_level = 2
            stress_text = "Moderate Stress"
            stress_score = 5
        else:
            stress_level = 1
            stress_text = "Low Stress"
            stress_score = 2
        
        # Adjust stress score based on exact polarity
        if stress_level == 3:
            stress_score = int(8 + (abs(combined_score) * 2))  # 8-10
        elif stress_level == 2:
            stress_score = int(4 + (0.5 - combined_score) * 6)  # 4-7
        else:
            stress_score = int(1 + (1 - combined_score) * 2)  # 1-3
        
        stress_score = max(1, min(10, stress_score))
        
        # Generate insights
        insights = self._generate_insights(
            polarity, subjectivity, keywords_found, stress_level
        )
        
        return {
            'success': True,
            'stress_level': stress_level,
            'stress_text': stress_text,
            'stress_score': stress_score,
            'sentiment': {
                'polarity': round(polarity, 3),
                'subjectivity': round(subjectivity, 3),
                'combined_score': round(combined_score, 3)
            },
            'keywords_found': keywords_found,
            'keyword_score': round(keyword_score, 3),
            'insights': insights,
            'word_count': len(cleaned_text.split())
        }
    
    def _generate_insights(self, polarity, subjectivity, keywords, stress_level):
        """Generate human-readable insights from analysis"""
        insights = []
        
        # Sentiment insight
        if polarity < -0.3:
            insights.append("Your text expresses predominantly negative emotions")
        elif polarity > 0.3:
            insights.append("Your text expresses positive emotions")
        else:
            insights.append("Your text shows mixed or neutral emotions")
        
        # Subjectivity insight
        if subjectivity > 0.6:
            insights.append("Your writing is quite personal and emotional")
        elif subjectivity < 0.3:
            insights.append("Your writing is fairly objective")
        
        # Keyword insights
        high_keywords = keywords['high_stress']
        low_keywords = keywords['low_stress']
        
        if high_keywords:
            insights.append(
                f"Detected stress indicators: {', '.join(high_keywords[:3])}"
            )
        
        if low_keywords:
            insights.append(
                f"Positive indicators found: {', '.join(low_keywords[:3])}"
            )
        
        # Overall insight
        if stress_level == 3:
            insights.append(
                "Consider reaching out to someone you trust or a mental health professional"
            )
        elif stress_level == 2:
            insights.append(
                "Taking regular breaks and practicing self-care may help"
            )
        else:
            insights.append(
                "Keep up the positive mindset!"
            )
        
        return insights


# Singleton instance
_analyzer = None

def get_text_analyzer():
    """Get or create singleton TextStressAnalyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = TextStressAnalyzer()
    return _analyzer


def analyze_text(text):
    """
    Convenience function for text stress analysis
    
    Args:
        text: Text string to analyze
    
    Returns:
        dict with stress_level, stress_text, sentiment, etc.
    """
    analyzer = get_text_analyzer()
    return analyzer.analyze(text)
