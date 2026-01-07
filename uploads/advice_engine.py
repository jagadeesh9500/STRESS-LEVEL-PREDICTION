# Factor-specific advice based on user inputs
FACTOR_ADVICE = {
    "anxiety_level": {
        "high": "Practice deep breathing exercises (4-7-8 technique) to calm anxiety. Try the box breathing method: inhale 4 seconds, hold 4 seconds, exhale 4 seconds.",
        "moderate": "Consider journaling your anxious thoughts. Writing them down can help process and release worry.",
        "low": "Great job managing your anxiety! Keep up your current coping strategies."
    },
    "self_esteem": {
        "high": "Challenge negative self-talk by listing 3 things you're proud of each day. You matter!",
        "moderate": "Try positive affirmations each morning. Celebrate small wins and acknowledge your efforts.",
        "low": "Your self-esteem is healthy! Continue to practice self-compassion."
    },
    "mental_health_history": {
        "high": "Given your mental health history, consider regular check-ins with a counselor. There's strength in seeking support.",
        "moderate": "Stay connected with your support system and maintain mental health routines that have helped before.",
        "low": "Continue to prioritize your mental wellness journey."
    },
    "depression": {
        "high": "Reach out to a mental health professional. Depression is treatable. Small steps like a 10-minute walk can help.",
        "moderate": "Maintain social connections and establish a daily routine. Exercise releases mood-boosting endorphins.",
        "low": "You're managing well. Stay engaged in activities that bring you joy."
    },
    "headache": {
        "high": "Frequent headaches may indicate stress or tension. Stay hydrated, take screen breaks, and consider relaxation exercises.",
        "moderate": "Monitor your headache triggers. Adequate sleep and reduced caffeine may help.",
        "low": "Good physical wellness! Keep maintaining your healthy habits."
    },
    "blood_pressure": {
        "high": "High blood pressure needs attention. Reduce salt intake, exercise regularly, and consider consulting a doctor.",
        "moderate": "Monitor your blood pressure regularly. Stress management and healthy eating can help.",
        "low": "Your cardiovascular health seems good. Keep up heart-healthy habits!"
    },
    "sleep_quality": {
        "high": "Improve sleep hygiene: set a consistent bedtime, avoid screens 1 hour before bed, and keep your room cool and dark.",
        "moderate": "Consider a calming bedtime routine. Limit caffeine after 2 PM and try relaxation techniques.",
        "low": "Excellent sleep habits! Quality sleep is foundation of mental wellness."
    },
    "breathing_problem": {
        "high": "Practice diaphragmatic breathing. If breathing issues persist, consult a healthcare provider.",
        "moderate": "Try the 4-7-8 breathing technique. Regular exercise can strengthen respiratory function.",
        "low": "Your breathing is well-regulated. Consider adding breathwork to maintain this."
    },
    "noise_level": {
        "high": "Consider noise-cancelling headphones or white noise machines. Find quiet spaces for focused work.",
        "moderate": "Create designated quiet time in your routine. Nature sounds or soft music may help.",
        "low": "Your environment seems peaceful. This supports mental clarity."
    },
    "living_conditions": {
        "high": "Improve your living space where possible: declutter, add plants, maximize natural light. Small changes make a difference.",
        "moderate": "Consider what aspects of your environment you can control and improve gradually.",
        "low": "Your living conditions support your wellbeing. Maintain a tidy, comfortable space."
    },
    "safety": {
        "high": "Your safety is paramount. If you feel unsafe, reach out to trusted people or local support services.",
        "moderate": "Identify what makes you feel unsafe and take steps to address those concerns.",
        "low": "Feeling safe is essential for mental health. Continue nurturing safe relationships."
    },
    "basic_needs": {
        "high": "Prioritize meeting your basic needs: nutrition, hydration, sleep. These are foundations of stress resilience.",
        "moderate": "Check if you're meeting all basic needs consistently. Regular meals and hydration help.",
        "low": "Your basic needs are being met. This provides a strong foundation for wellbeing."
    },
    "academic_performance": {
        "high": "Break large tasks into smaller, manageable steps. Consider tutoring or study groups for support.",
        "moderate": "Use effective study techniques like spaced repetition. Don't be afraid to ask for help.",
        "low": "Great academic progress! Balance studies with rest and recreation."
    },
    "study_load": {
        "high": "Prioritize tasks using the Eisenhower Matrix. Learn to say no to non-essential commitments.",
        "moderate": "Create a realistic study schedule with built-in breaks. Use the Pomodoro technique.",
        "low": "Your workload is manageable. Maintain this balance for sustained success."
    },
    "teacher_student_relationship": {
        "high": "Try to build rapport with teachers through office hours or after-class conversations. Communication helps.",
        "moderate": "Don't hesitate to ask questions. Most teachers appreciate engaged students.",
        "low": "Good relationships with educators support learning and reduce stress."
    },
    "future_career_concerns": {
        "high": "Focus on what you can control today. Explore career counseling services and talk to mentors in your field of interest.",
        "moderate": "Create a loose plan but stay flexible. Many successful people changed paths multiple times.",
        "low": "You have a healthy perspective on your future. Continue exploring your interests."
    },
    "social_support": {
        "high": "Reach out to at least one person today. Join clubs, groups, or online communities that match your interests.",
        "moderate": "Nurture your existing relationships. Quality connections matter more than quantity.",
        "low": "Strong social support is protective against stress. Keep investing in relationships."
    },
    "peer_pressure": {
        "high": "Practice saying no assertively. Surround yourself with people who respect your choices.",
        "moderate": "Remember your values when facing pressure. It's okay to take time before making decisions.",
        "low": "You're handling peer dynamics well. Stay true to yourself."
    },
    "extracurricular_activities": {
        "high": "Evaluate if you're overcommitted. It's okay to drop activities that drain you.",
        "moderate": "Find balance between activities and rest. Choose activities that genuinely energize you.",
        "low": "You have a healthy activity balance. Hobbies are important for stress relief."
    },
    "bullying": {
        "high": "Bullying is never okay. Report it to trusted authorities. You deserve to feel safe and respected.",
        "moderate": "Document any incidents and speak to someone you trust. You don't have to handle this alone.",
        "low": "You're in a supportive environment. Continue to be kind to others and yourself."
    }
}

# Emergency resources for high stress
EMERGENCY_RESOURCES = {
    "india": {
        "name": "iCall (TISS)",
        "number": "9152987821",
        "description": "Professional counseling support"
    },
    "vandrevala": {
        "name": "Vandrevala Foundation",
        "number": "1860-2662-345",
        "description": "24/7 mental health helpline"
    },
    "nimhans": {
        "name": "NIMHANS Helpline",
        "number": "080-46110007",
        "description": "National mental health support"
    }
}

def get_factor_level(feature, value, inverse_features):
    """Determine if a factor indicates high, moderate, or low stress"""
    if feature in inverse_features:
        # For inverse features, low value = high stress
        if value <= 3:
            return "high"
        elif value <= 6:
            return "moderate"
        else:
            return "low"
    else:
        # For normal features, high value = high stress
        if value >= 7:
            return "high"
        elif value >= 4:
            return "moderate"
        else:
            return "low"

def generate_advice(stress_level, user_inputs=None, top_factors=None, inverse_features=None):
    """Generate personalized advice based on stress level and contributing factors"""
    
    advice = []
    
    # Add stress level summary
    if stress_level == 3:
        advice.append("⚠️ High stress detected - prioritize self-care and consider professional support")
    elif stress_level == 2:
        advice.append("📊 Moderate stress level - some areas need attention")
    else:
        advice.append("✅ Low stress level - you're managing well!")
    
    # If we have user inputs and top factors, provide factor-specific advice
    if user_inputs and top_factors and inverse_features:
        advice.append("")  # Add spacing
        advice.append("📌 Focus Areas Based on Your Assessment:")
        
        # Get advice for top contributing factors
        for factor, importance in top_factors[:3]:
            if factor in user_inputs and factor in FACTOR_ADVICE:
                value = user_inputs[factor]
                level = get_factor_level(factor, value, inverse_features)
                factor_name = factor.replace('_', ' ').title()
                
                if level in ["high", "moderate"]:
                    advice.append(f"• {factor_name}: {FACTOR_ADVICE[factor][level]}")
        
        # Add advice for any high-stress factors not in top 3
        high_stress_factors = []
        for feature, value in user_inputs.items():
            level = get_factor_level(feature, value, inverse_features)
            if level == "high" and feature not in [f[0] for f in top_factors[:3]]:
                high_stress_factors.append((feature, value))
        
        if high_stress_factors:
            advice.append("")
            advice.append("💡 Additional Recommendations:")
            for factor, value in high_stress_factors[:2]:  # Limit to 2 more
                if factor in FACTOR_ADVICE:
                    factor_name = factor.replace('_', ' ').title()
                    advice.append(f"• {factor_name}: {FACTOR_ADVICE[factor]['high']}")
    
    # Add general tips based on stress level
    advice.append("")
    if stress_level == 3:
        advice.append("🧘 General Tips for High Stress:")
        advice.append("• Take 5 minutes for deep breathing right now")
        advice.append("• Reach out to someone you trust today")
        advice.append("• Consider speaking with a mental health professional")
    elif stress_level == 2:
        advice.append("💪 General Tips for Moderate Stress:")
        advice.append("• Schedule regular breaks throughout your day")
        advice.append("• Engage in 20-30 minutes of physical activity")
        advice.append("• Practice mindfulness or meditation for 10 minutes daily")
    else:
        advice.append("🌟 Tips to Maintain Low Stress:")
        advice.append("• Continue your healthy routines")
        advice.append("• Stay connected with your support network")
        advice.append("• Celebrate your wins, big and small")
    
    return advice

def get_emergency_resources():
    """Return emergency mental health resources"""
    return EMERGENCY_RESOURCES
