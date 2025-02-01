import spacy
import logging
from typing import List, Dict, Set
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nlp_singleton import get_nlp 

class TopicAnalyzer:
    def __init__(self):
        self.nlp = get_nlp()
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.topic_hierarchy: Dict[str, Set[str]] = {}
        
    def extract_keywords(self, text: str, question: str = "", answer: str = "") -> List[str]:
        """Extract keywords with PDF context awareness."""
        self._build_topic_hierarchy(text)
        qa_keywords = self._extract_qa_keywords(question, answer)
        doc_keywords = self._extract_document_keywords(text)
        combined_keywords = self._combine_and_rank_keywords(qa_keywords, doc_keywords, text)
        return combined_keywords[:5]
    
    def _build_topic_hierarchy(self, text: str):
        """Build a hierarchical topic structure from the document."""
        doc = self.nlp(text)
        
        # Extract main topics and subtopics
        main_topics = set()
        subtopics = {}
        
        for sent in doc.sents:
            # Find main topic in the sentence
            main_topic = None
            for token in sent:
                if token.dep_ in ['nsubj', 'ROOT'] and token.pos_ in ['NOUN', 'PROPN']:
                    main_topic = token.text.lower()
                    main_topics.add(main_topic)
                    break
            
            # Find related subtopics
            if main_topic:
                if main_topic not in subtopics:
                    subtopics[main_topic] = set()
                
                for token in sent:
                    if (token.dep_ in ['dobj', 'pobj'] and 
                        token.pos_ in ['NOUN', 'PROPN']):
                        subtopics[main_topic].add(token.text.lower())
        
        self.topic_hierarchy = subtopics
    
    def _extract_qa_keywords(self, question: str, answer: str) -> List[str]:
        """Extract keywords from question and answer with weights."""
        combined = f"{question} {answer}"
        doc = self.nlp(combined)
        
        keywords = []
        for token in doc:
            # Weight different parts of speech
            if token.pos_ in ['NOUN', 'PROPN']:
                keywords.append((token.text.lower(), 1.0))
            elif token.pos_ in ['VERB']:
                keywords.append((token.text.lower(), 0.7))
            elif token.pos_ in ['ADJ']:
                keywords.append((token.text.lower(), 0.5))
        
        return [k[0] for k in sorted(keywords, key=lambda x: x[1], reverse=True)]
    
    def _extract_document_keywords(self, text: str) -> List[str]:
        """Extract keywords from the document using TF-IDF."""
        # Split text into sentences for TF-IDF
        sentences = [sent.text for sent in self.nlp(text).sents]
        if not sentences:
            return []
            
        try:
            # Calculate TF-IDF
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            feature_names = self.tfidf.get_feature_names_out()
            
            # Get important terms based on TF-IDF scores
            importance = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
            indices = importance.argsort()[-20:][::-1]  # Top 20 terms
            
            return [feature_names[i] for i in indices]
            
        except Exception as e:
            logging.warning(f"TF-IDF extraction failed: {e}")
            return []
    
    def _combine_and_rank_keywords(
        self, qa_keywords: List[str], 
        doc_keywords: List[str], 
        context: str
    ) -> List[str]:
        """Combine and rank keywords based on multiple factors."""
        keyword_scores = Counter()
        
        # Score based on presence in QA
        for i, keyword in enumerate(qa_keywords):
            keyword_scores[keyword] += 1.0 / (i + 1)
        
        # Score based on document importance
        for i, keyword in enumerate(doc_keywords):
            keyword_scores[keyword] += 0.8 / (i + 1)
        
        # Score based on topic hierarchy
        for main_topic, subtopics in self.topic_hierarchy.items():
            if main_topic in keyword_scores:
                keyword_scores[main_topic] *= 1.2
            for subtopic in subtopics:
                if subtopic in keyword_scores:
                    keyword_scores[subtopic] *= 1.1
        
        # Get final ranked keywords
        return [k for k, v in keyword_scores.most_common(10)]