from typing import Dict, List, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter

class PerformanceAnalyzer:
    def __init__(self):
        self._cache = {}
        
    def generate_report_data(self, session_data: Dict) -> Dict:
        """Generate comprehensive report data with improved weak points analysis."""
        cache_key = f"{hash(frozenset(str(session_data.items()).encode()))}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        metrics = self._calculate_core_metrics(session_data)
        charts = self._generate_charts(session_data)
        improvements = self._analyze_improvement_areas(session_data)
        weak_points = self._analyze_weak_points(session_data)
        learning_path = self._generate_learning_path(weak_points)
        
        report_data = {
            "metrics": metrics,
            "charts": charts,
            "improvements": improvements,
            "weak_points": weak_points,
            "learning_path": learning_path
        }
        
        self._cache[cache_key] = report_data
        return report_data
    
    def _calculate_core_metrics(self, data: Dict) -> Dict:
        """Calculate detailed performance metrics."""
        total_questions = data["total_questions"]
        if total_questions == 0:
            return {
                "accuracy": 0,
                "total_score": 0,
                "avg_response_time": 0,
                "questions_completed": 0,
                "performance_rating": "N/A"
            }
            
        accuracy = (data["correct_answers"] / total_questions * 100)
        avg_time = np.mean(data["question_times"]) if data["question_times"] else 0
        
        # Calculate performance rating
        if accuracy >= 90:
            rating = "Excellent"
        elif accuracy >= 75:
            rating = "Good"
        elif accuracy >= 60:
            rating = "Fair"
        else:
            rating = "Needs Improvement"
        
        return {
            "accuracy": round(accuracy, 1),
            "total_score": data["score"],
            "avg_response_time": round(avg_time, 1),
            "questions_completed": total_questions,
            "performance_rating": rating
        }
    
    def _analyze_weak_points(self, data: Dict) -> Dict:
        """Detailed analysis of weak points and patterns."""
        if not data["question_history"]:
            return {"topics": [], "patterns": {}, "recommendations": []}
            
        # Analyze incorrect answers by topic
        topic_errors = Counter(data["weak_topics"])
        weak_topics = [
            {
                "topic": topic,
                "count": count,
                "percentage": (count / len(data["question_history"])) * 100
            }
            for topic, count in topic_errors.most_common()
        ]
        
        # Analyze patterns in incorrect answers
        patterns = {
            "time_related": self._analyze_time_based_errors(data),
            "difficulty_related": self._analyze_difficulty_based_errors(data),
            "consecutive_errors": self._analyze_consecutive_errors(data)
        }
        
        # Generate specific recommendations
        recommendations = self._generate_recommendations(weak_topics, patterns)
        
        return {
            "topics": weak_topics,
            "patterns": patterns,
            "recommendations": recommendations
        }
    
    def _analyze_time_based_errors(self, data: Dict) -> Dict:
        """Analyze correlation between response time and errors."""
        if not data["question_times"]:
            return {"found": False}
            
        times = np.array(data["question_times"])
        correct = np.array([1 if i < data["correct_answers"] else 0 
                          for i in range(len(times))])
        
        # Check if longer times correlate with incorrect answers
        slow_responses = times > np.median(times)
        errors_in_slow = np.sum(correct[slow_responses] == 0)
        errors_in_fast = np.sum(correct[~slow_responses] == 0)
        
        return {
            "found": True,
            "slow_response_errors": int(errors_in_slow),
            "fast_response_errors": int(errors_in_fast),
            "time_impact": errors_in_slow > errors_in_fast
        }
    
    def _analyze_difficulty_based_errors(self, data: Dict) -> Dict:
        """Analyze error patterns based on question difficulty."""
        difficulty_errors = {1: 0, 2: 0, 3: 0}
        difficulty_totals = Counter(data["difficulty_stats"])
        
        for diff, count in difficulty_totals.items():
            if count > 0:
                errors = sum(1 for q in data["question_history"] 
                           if q.get("difficulty") == diff and not q.get("correct"))
                difficulty_errors[diff] = (errors / count) * 100
                
        return {
            "error_rates": difficulty_errors,
            "challenging_level": max(difficulty_errors.items(), key=lambda x: x[1])[0]
        }
    
    def _analyze_consecutive_errors(self, data: Dict) -> Dict:
        """Analyze patterns of consecutive incorrect answers."""
        if not data["question_history"]:
            return {"found": False}
            
        consecutive_errors = 0
        max_consecutive = 0
        current_streak = 0
        
        for q in data["question_history"]:
            if not q.get("correct"):
                current_streak += 1
                if current_streak > 1:
                    consecutive_errors += 1
                max_consecutive = max(max_consecutive, current_streak)
            else:
                current_streak = 0
                
        return {
            "found": consecutive_errors > 0,
            "count": consecutive_errors,
            "max_streak": max_consecutive
        }
    
    def _generate_recommendations(self, weak_topics: List[Dict], patterns: Dict) -> List[str]:
        """Generate specific, actionable recommendations based on analysis."""
        recommendations = []
        
        # Topic-based recommendations
        for topic in weak_topics[:3]:  # Top 3 weak topics
            if topic["percentage"] > 50:
                recommendations.append(f"Review {topic['topic']} fundamentals - showing significant weakness")
            elif topic["percentage"] > 30:
                recommendations.append(f"Practice more questions on {topic['topic']}")
        
        # Time-based recommendations
        if patterns["time_related"]["found"]:
            if patterns["time_related"]["time_impact"]:
                recommendations.append("Work on time management - accuracy decreases with longer response times")
            else:
                recommendations.append("Consider taking more time to answer - quick responses may be hurting accuracy")
        
        # Difficulty-based recommendations
        challenging_level = patterns["difficulty_related"]["challenging_level"]
        if challenging_level == 3:
            recommendations.append("Focus on building advanced topic understanding before tackling harder questions")
        elif challenging_level == 2:
            recommendations.append("Practice intermediate-level questions to build confidence")
        
        # Consecutive error recommendations
        if patterns["consecutive_errors"]["found"] and patterns["consecutive_errors"]["max_streak"] > 2:
            recommendations.append("Take short breaks when encountering multiple difficult questions")
        
        return recommendations
    
    def _generate_learning_path(self, weak_points: Dict) -> List[Dict]:
        """Generate a personalized learning path based on weak points."""
        if not weak_points["topics"]:
            return []
            
        learning_path = []
        for topic in weak_points["topics"]:
            if topic["percentage"] > 40:
                learning_path.append({
                    "topic": topic["topic"],
                    "priority": "High",
                    "focus_areas": [
                        "Review fundamental concepts",
                        "Practice basic problems",
                        "Gradually increase difficulty"
                    ]
                })
            elif topic["percentage"] > 20:
                learning_path.append({
                    "topic": topic["topic"],
                    "priority": "Medium",
                    "focus_areas": [
                        "Practice targeted problems",
                        "Identify specific challenge areas"
                    ]
                })
        
        return learning_path

