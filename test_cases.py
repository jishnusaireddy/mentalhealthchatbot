#!/usr/bin/env python3
"""
Test cases for the Mental Health Conversational Agent
"""

import unittest
import requests
import json
import time
from datetime import datetime

class TestMentalHealthAgent(unittest.TestCase):
    """Test cases for the mental health conversational agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5000/api"
        self.test_session_id = f"test_session_{int(time.time())}"
        self.test_user_id = "test_user"
        
    def test_chat_endpoint_basic(self):
        """Test basic chat functionality"""
        url = f"{self.base_url}/chat"
        payload = {
            "message": "Hello",
            "session_id": self.test_session_id,
            "user_id": self.test_user_id
        }
        
        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("agent_response", data)
        self.assertIn("intent", data)
        self.assertIn("sentiment", data)
        self.assertEqual(data["intent"], "greeting")
        
    def test_intent_detection_anxiety(self):
        """Test intent detection for anxiety-related messages"""
        url = f"{self.base_url}/chat"
        payload = {
            "message": "I'm feeling very anxious about my presentation tomorrow",
            "session_id": self.test_session_id,
            "user_id": self.test_user_id
        }
        
        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["intent"], "anxiety")
        self.assertIn("coping strategy", data["agent_response"].lower())
        
    def test_intent_detection_depression(self):
        """Test intent detection for depression-related messages"""
        url = f"{self.base_url}/chat"
        payload = {
            "message": "I've been feeling really sad and hopeless lately",
            "session_id": self.test_session_id,
            "user_id": self.test_user_id
        }
        
        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["intent"], "depression")
        
    def test_crisis_detection(self):
        """Test crisis detection and escalation"""
        url = f"{self.base_url}/chat"
        payload = {
            "message": "I want to hurt myself",
            "session_id": self.test_session_id,
            "user_id": self.test_user_id
        }
        
        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["intent"], "crisis")
        self.assertEqual(data["action_required"], "crisis_escalation")
        self.assertIn("resources", data)
        self.assertTrue(len(data["resources"]) > 0)
        
    def test_sentiment_analysis_positive(self):
        """Test sentiment analysis for positive messages"""
        url = f"{self.base_url}/chat"
        payload = {
            "message": "I'm feeling great today and very happy",
            "session_id": self.test_session_id,
            "user_id": self.test_user_id
        }
        
        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["sentiment"], "positive")
        
    def test_sentiment_analysis_negative(self):
        """Test sentiment analysis for negative messages"""
        url = f"{self.base_url}/chat"
        payload = {
            "message": "I'm feeling terrible and everything is awful",
            "session_id": self.test_session_id,
            "user_id": self.test_user_id
        }
        
        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["sentiment"], "negative")
        
    def test_conversation_history(self):
        """Test conversation history retrieval"""
        # First, send a message
        chat_url = f"{self.base_url}/chat"
        payload = {
            "message": "Hello, this is a test message",
            "session_id": self.test_session_id,
            "user_id": self.test_user_id
        }
        
        chat_response = requests.post(chat_url, json=payload)
        self.assertEqual(chat_response.status_code, 200)
        
        # Then, retrieve history
        history_url = f"{self.base_url}/history/{self.test_session_id}"
        history_response = requests.get(history_url)
        self.assertEqual(history_response.status_code, 200)
        
        history_data = history_response.json()
        self.assertIn("history", history_data)
        self.assertTrue(len(history_data["history"]) >= 2)  # User message + agent response
        
    def test_resources_endpoint(self):
        """Test resources endpoint"""
        url = f"{self.base_url}/resources"
        
        # Test general resources
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("resources", data)
        self.assertTrue(len(data["resources"]) > 0)
        
        # Test crisis resources
        response = requests.get(url, params={"category": "crisis"})
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("resources", data)
        self.assertTrue(len(data["resources"]) > 0)
        
    def test_feedback_endpoint(self):
        """Test feedback submission"""
        url = f"{self.base_url}/feedback"
        payload = {
            "user_id": self.test_user_id,
            "message_id": "test_message_123",
            "rating": 5,
            "comment": "Very helpful response"
        }
        
        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertEqual(data["status"], "success")
        
    def test_empty_message_handling(self):
        """Test handling of empty messages"""
        url = f"{self.base_url}/chat"
        payload = {
            "message": "",
            "session_id": self.test_session_id,
            "user_id": self.test_user_id
        }
        
        response = requests.post(url, json=payload)
        self.assertEqual(response.status_code, 400)
        
    def test_malformed_request(self):
        """Test handling of malformed requests"""
        url = f"{self.base_url}/chat"
        payload = {
            "invalid_field": "test"
        }
        
        response = requests.post(url, json=payload)
        # Should handle gracefully, either with 400 or default values
        self.assertIn(response.status_code, [200, 400])

class TestPerformance(unittest.TestCase):
    """Performance tests for the mental health agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5000/api"
        self.test_session_id = f"perf_test_{int(time.time())}"
        self.test_user_id = "perf_test_user"
        
    def test_response_time(self):
        """Test response time for chat endpoint"""
        url = f"{self.base_url}/chat"
        payload = {
            "message": "Hello, how are you?",
            "session_id": self.test_session_id,
            "user_id": self.test_user_id
        }
        
        start_time = time.time()
        response = requests.post(url, json=payload)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        self.assertLess(response_time, 5.0)  # Response should be under 5 seconds
        
    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import threading
        
        url = f"{self.base_url}/chat"
        results = []
        
        def send_request(message_id):
            payload = {
                "message": f"Test message {message_id}",
                "session_id": f"{self.test_session_id}_{message_id}",
                "user_id": self.test_user_id
            }
            
            response = requests.post(url, json=payload)
            results.append(response.status_code)
        
        # Create 5 concurrent threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=send_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        self.assertEqual(len(results), 5)
        for status_code in results:
            self.assertEqual(status_code, 200)

def run_evaluation():
    """Run evaluation metrics on the agent's performance"""
    print("Running Mental Health Agent Evaluation...")
    
    # Test cases for evaluation
    test_cases = [
        {
            "input": "Hello",
            "expected_intent": "greeting",
            "expected_sentiment": "neutral"
        },
        {
            "input": "I'm feeling anxious about my job interview",
            "expected_intent": "anxiety",
            "expected_sentiment": "negative"
        },
        {
            "input": "I've been feeling depressed lately",
            "expected_intent": "depression",
            "expected_sentiment": "negative"
        },
        {
            "input": "I want to hurt myself",
            "expected_intent": "crisis",
            "expected_sentiment": "negative"
        },
        {
            "input": "I'm having a great day!",
            "expected_intent": "general_support",
            "expected_sentiment": "positive"
        }
    ]
    
    base_url = "http://localhost:5000/api"
    session_id = f"eval_session_{int(time.time())}"
    
    correct_intent = 0
    correct_sentiment = 0
    total_tests = len(test_cases)
    
    print(f"\nTesting {total_tests} cases...")
    
    for i, test_case in enumerate(test_cases):
        payload = {
            "message": test_case["input"],
            "session_id": session_id,
            "user_id": "eval_user"
        }
        
        try:
            response = requests.post(f"{base_url}/chat", json=payload)
            if response.status_code == 200:
                data = response.json()
                
                # Check intent accuracy
                if data["intent"] == test_case["expected_intent"]:
                    correct_intent += 1
                    intent_result = "✓"
                else:
                    intent_result = "✗"
                
                # Check sentiment accuracy
                if data["sentiment"] == test_case["expected_sentiment"]:
                    correct_sentiment += 1
                    sentiment_result = "✓"
                else:
                    sentiment_result = "✗"
                
                print(f"Test {i+1}: Intent {intent_result} ({data['intent']}) | Sentiment {sentiment_result} ({data['sentiment']})")
            else:
                print(f"Test {i+1}: Failed with status {response.status_code}")
                
        except Exception as e:
            print(f"Test {i+1}: Error - {str(e)}")
    
    # Calculate accuracy
    intent_accuracy = (correct_intent / total_tests) * 100
    sentiment_accuracy = (correct_sentiment / total_tests) * 100
    
    print(f"\n=== Evaluation Results ===")
    print(f"Intent Detection Accuracy: {intent_accuracy:.1f}% ({correct_intent}/{total_tests})")
    print(f"Sentiment Analysis Accuracy: {sentiment_accuracy:.1f}% ({correct_sentiment}/{total_tests})")
    print(f"Overall Performance: {(intent_accuracy + sentiment_accuracy) / 2:.1f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        run_evaluation()
    else:
        unittest.main()

