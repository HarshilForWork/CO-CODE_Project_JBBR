# performance_analyzer.py
from typing import Dict, List
import numpy as np
from collections import Counter
from nlp_singleton import get_nlp

class PerformanceAnalyzer:
    def __init__(self, llm):
        self.llm = llm["model"]
        self.nlp = get_nlp()
        self.report_llm = llm["report_model"]  # Use specific model for reports
        
    def generate_performance_insights(self, performance_data: Dict) -> str:
        """Generate detailed performance insights using report-specific LLM."""
        metrics = self._prepare_metrics(performance_data)
        prompt = f"""
        As an educational assessment expert, analyze the following quiz performance data and provide detailed insights:

        Performance Metrics:
        - Overall Accuracy: {metrics['accuracy']}%
        - Total Questions Attempted: {metrics['total_questions']}
        - Average Response Time: {metrics['avg_response_time']:.2f} seconds
        - Difficulty Distribution: {metrics['difficulty_dist']}
        
        Weak Topics:
        {metrics['weak_topics_str']}
        
        Question Performance:
        {metrics['question_performance']}
        
        Provide a detailed analysis that includes:
        1. Overall performance assessment
        2. Specific strengths and areas for improvement
        3. Learning patterns and trends
        4. Actionable recommendations for improvement
        5. Suggested focus areas based on weak topics
        
        Format the response in clear sections with bullet points for key findings.
        """
        
        try:
            # Use the report-specific model
            analysis = self.report_llm.invoke(prompt)
            return analysis
        except Exception as e:
            return self._generate_fallback_analysis(metrics)
    
    def _prepare_metrics(self, data: Dict) -> Dict:
        """Prepare and format performance metrics for analysis."""
        # Calculate accuracy
        accuracy = (data['correct_answers'] / data['total_questions'] * 100) if data['total_questions'] > 0 else 0
        
        # Calculate average response time
        avg_response_time = np.mean(data['question_times']) if data['question_times'] else 0
        
        # Format difficulty distribution using the actual keys from difficulty_stats
        diff_dist = []
        for level, count in data['difficulty_stats'].items():
            diff_dist.append(f"{level}: {count}")
        diff_str = ", ".join(diff_dist)
        
        # Format weak topics
        weak_topics = Counter(data['weak_topics']).most_common(5)
        weak_topics_str = "\n".join([f"- {topic}: {count} incorrect" for topic, count in weak_topics])
        
        # Format question performance
        question_perf = []
        for i, time in enumerate(data['question_times']):
            status = "Correct" if i < data['correct_answers'] else "Incorrect"
            question_perf.append(f"Q{i+1}: {status} ({time:.2f}s)")
        
        return {
            'accuracy': accuracy,
            'total_questions': data['total_questions'],
            'avg_response_time': avg_response_time,
            'difficulty_dist': diff_str,
            'weak_topics_str': weak_topics_str,
            'question_performance': "\n".join(question_perf)
        }
    
    def _generate_fallback_analysis(self, metrics: Dict) -> str:
        """Generate basic analysis if LLM fails."""
        performance_level = "Excellent" if metrics['accuracy'] >= 80 else "Good" if metrics['accuracy'] >= 60 else "Needs Improvement"
        
        return f"""
        Performance Analysis:
        
        Overall Assessment:
        • Performance Level: {performance_level}
        • Accuracy: {metrics['accuracy']:.1f}%
        • Average Response Time: {metrics['avg_response_time']:.1f} seconds
        
        Key Findings:
        • Question Distribution: {metrics['difficulty_dist']}
        
        Areas for Improvement:
        {metrics['weak_topics_str']}
        
        Recommendations:
        • Focus on reviewing the identified weak topics
        • Practice with more questions in challenging areas
        • Work on improving response time while maintaining accuracy
        """

    def optimize_mcq_generation(self, context: str) -> Dict:
        """Optimize context for faster MCQ generation."""
        # Implement preprocessing optimizations
        doc = self.nlp(context)
        
        # Extract key sentences based on importance
        important_sentences = []
        for sent in doc.sents:
            # Check for key features indicating importance
            has_entities = any(ent.label_ in ['ORG', 'PERSON', 'GPE', 'DATE', 'NORP'] for ent in sent.ents)
            has_keywords = any(token.pos_ in ['NOUN', 'VERB'] and not token.is_stop for token in sent)
            
            if has_entities or has_keywords:
                important_sentences.append(sent.text)
        
        # Create optimized context
        optimized_context = " ".join(important_sentences)
        
        return {
            "optimized_context": optimized_context,
            "key_entities": [ent.text for ent in doc.ents],
            "context_length": len(optimized_context.split())
        }