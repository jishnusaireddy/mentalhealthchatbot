from flask import Blueprint, jsonify, request
from datetime import datetime
import uuid
import re
import random

chat_bp = Blueprint('chat', __name__)

# Simple in-memory storage for demonstration (in production, use a proper database)
conversations = {}
user_sessions = {}

# Crisis keywords for basic detection
CRISIS_KEYWORDS = [
    'suicide', 'kill myself', 'end my life', 'want to die', 'hurt myself',
    'self harm', 'cutting', 'overdose', 'jump off', 'hanging'
]

# Supportive responses for different intents
SUPPORTIVE_RESPONSES = {
    'greeting': [
        "Hello! I'm here to listen and support you. How are you feeling today?",
        "Hi there! I'm glad you're here. What's on your mind?",
        "Welcome! I'm here to help. How can I support you today?"
    ],
    'distress': [
        "I hear that you're going through a difficult time. Your feelings are valid, and I'm here to support you.",
        "It sounds like you're experiencing some challenging emotions. Would you like to talk about what's troubling you?",
        "I understand this is hard for you. Remember that seeking support is a sign of strength."
    ],
    'anxiety': [
        "Anxiety can feel overwhelming, but you're not alone. Let's try some breathing exercises together.",
        "I understand anxiety can be very difficult. Would you like to explore some coping strategies?",
        "Feeling anxious is a normal human experience. Let's work through this together."
    ],
    'depression': [
        "Depression can make everything feel heavy. I want you to know that your feelings matter and help is available.",
        "I hear you, and I want you to know that depression is treatable. You don't have to face this alone.",
        "Thank you for sharing with me. Depression affects many people, and there are ways to feel better."
    ],
    'general_support': [
        "I'm here to listen without judgment. What would be most helpful for you right now?",
        "Your mental health matters. How can I best support you today?",
        "I appreciate you reaching out. What's been on your mind lately?"
    ]
}

COPING_STRATEGIES = [
    "Try the 4-7-8 breathing technique: Breathe in for 4 counts, hold for 7, exhale for 8.",
    "Practice grounding: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
    "Consider writing down your thoughts in a journal to help process your emotions.",
    "Gentle physical activity like a short walk can help improve mood.",
    "Reach out to a trusted friend or family member for support."
]

def detect_intent(message):
    """Simple intent detection based on keywords"""
    message_lower = message.lower()
    
    # Check for crisis indicators
    if any(keyword in message_lower for keyword in CRISIS_KEYWORDS):
        return 'crisis'
    
    # Check for specific mental health topics
    if any(word in message_lower for word in ['anxious', 'anxiety', 'worried', 'panic']):
        return 'anxiety'
    
    if any(word in message_lower for word in ['depressed', 'depression', 'sad', 'hopeless', 'empty']):
        return 'depression'
    
    if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
        return 'greeting'
    
    if any(word in message_lower for word in ['stressed', 'overwhelmed', 'difficult', 'hard time', 'struggling']):
        return 'distress'
    
    return 'general_support'

def analyze_sentiment(message):
    """Simple sentiment analysis based on keywords"""
    positive_words = ['good', 'great', 'happy', 'better', 'fine', 'okay', 'well']
    negative_words = ['bad', 'terrible', 'awful', 'worse', 'sad', 'angry', 'frustrated']
    
    message_lower = message.lower()
    positive_count = sum(1 for word in positive_words if word in message_lower)
    negative_count = sum(1 for word in negative_words if word in message_lower)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

def generate_response(intent, message):
    """Generate appropriate response based on intent"""
    if intent == 'crisis':
        return {
            'response': "I'm very concerned about what you've shared. Please reach out to a crisis helpline immediately: National Suicide Prevention Lifeline: 988 or 1-800-273-8255. You can also text HOME to 741741 for the Crisis Text Line. Your life has value, and help is available.",
            'action_required': 'crisis_escalation',
            'resources': [
                {'name': 'National Suicide Prevention Lifeline', 'phone': '988'},
                {'name': 'Crisis Text Line', 'text': 'HOME to 741741'},
                {'name': 'Emergency Services', 'phone': '911'}
            ]
        }
    
    responses = SUPPORTIVE_RESPONSES.get(intent, SUPPORTIVE_RESPONSES['general_support'])
    base_response = random.choice(responses)
    
    # Add coping strategy for certain intents
    if intent in ['anxiety', 'distress', 'depression']:
        coping_strategy = random.choice(COPING_STRATEGIES)
        base_response += f"\n\nHere's a coping strategy you might try: {coping_strategy}"
    
    return {
        'response': base_response,
        'action_required': 'none',
        'resources': []
    }

@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_id = data.get('user_id', str(uuid.uuid4()))
        message = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Initialize conversation history if needed
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Store user message
        user_message = {
            'sender': 'user',
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        conversations[session_id].append(user_message)
        
        # Analyze message
        intent = detect_intent(message)
        sentiment = analyze_sentiment(message)
        
        # Generate response
        response_data = generate_response(intent, message)
        
        # Store agent response
        agent_message = {
            'sender': 'agent',
            'message': response_data['response'],
            'timestamp': datetime.now().isoformat(),
            'intent': intent,
            'sentiment': sentiment
        }
        conversations[session_id].append(agent_message)
        
        return jsonify({
            'agent_response': response_data['response'],
            'sentiment': sentiment,
            'intent': intent,
            'action_required': response_data['action_required'],
            'resources': response_data.get('resources', []),
            'session_id': session_id,
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/history/<session_id>', methods=['GET'])
def get_history(session_id):
    try:
        history = conversations.get(session_id, [])
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        user_id = data.get('user_id')
        message_id = data.get('message_id')
        rating = data.get('rating')
        comment = data.get('comment', '')
        
        # In a real application, store this feedback in a database
        feedback_data = {
            'user_id': user_id,
            'message_id': message_id,
            'rating': rating,
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        
        # For now, just log it (in production, save to database)
        print(f"Feedback received: {feedback_data}")
        
        return jsonify({'status': 'success', 'message': 'Feedback received'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/resources', methods=['GET'])
def get_resources():
    try:
        category = request.args.get('category', 'general')
        
        resources = {
            'crisis': [
                {'name': 'National Suicide Prevention Lifeline', 'phone': '988', 'description': '24/7 crisis support'},
                {'name': 'Crisis Text Line', 'text': 'HOME to 741741', 'description': 'Text-based crisis support'},
                {'name': 'Emergency Services', 'phone': '911', 'description': 'Immediate emergency assistance'}
            ],
            'general': [
                {'name': 'National Alliance on Mental Illness (NAMI)', 'url': 'https://nami.org', 'description': 'Mental health education and support'},
                {'name': 'Mental Health America', 'url': 'https://mhanational.org', 'description': 'Mental health resources and screening tools'},
                {'name': 'Psychology Today', 'url': 'https://psychologytoday.com', 'description': 'Find therapists and mental health professionals'}
            ],
            'anxiety': [
                {'name': 'Anxiety and Depression Association of America', 'url': 'https://adaa.org', 'description': 'Resources for anxiety disorders'},
                {'name': 'Calm App', 'url': 'https://calm.com', 'description': 'Meditation and relaxation exercises'},
                {'name': 'Headspace', 'url': 'https://headspace.com', 'description': 'Mindfulness and meditation'}
            ]
        }
        
        return jsonify({'resources': resources.get(category, resources['general'])})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

