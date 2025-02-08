import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from keybert import KeyBERT
import re

class TopicSegmenter:
    def __init__(self, window_size=3, similarity_threshold=0.2, context_size=2, min_segment_size=3, topic_similarity_threshold=0.3):
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.context_size = context_size
        self.min_segment_size = min_segment_size
        self.topic_similarity_threshold = topic_similarity_threshold
        self.keyword_extractor = KeyBERT()
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        
        self.segment_vectorizer = TfidfVectorizer(
            stop_words='english',
            min_df=1,
            max_df=1.0,
            ngram_range=(1, 2)
        )
        
        self.lemmatizer = WordNetLemmatizer()
        self.topic_history = []
        self.fitted_vectorizer = None
        
    def preprocess_text(self, text):
        sentences = sent_tokenize(text)
        original_sentences = sentences.copy()
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = re.sub(r'^[^:]+:', '', sentence).strip()
            sentence = re.sub(r'[^\w\s]', '', sentence.lower())
            words = word_tokenize(sentence)
            lemmatized = [self.lemmatizer.lemmatize(word) for word in words]
            cleaned_sentence = ' '.join(lemmatized)
            cleaned_sentences.append(cleaned_sentence)
            
        return cleaned_sentences, original_sentences
    
    def get_topic_fingerprint(self, segment_text):
        cleaned_text = ' '.join([re.sub(r'^[^:]+:', '', sent).strip() for sent in segment_text])
        
        if self.fitted_vectorizer is None:
            self.fitted_vectorizer = self.segment_vectorizer.fit([cleaned_text])
            tfidf_matrix = self.fitted_vectorizer.transform([cleaned_text])
        else:
            tfidf_matrix = self.fitted_vectorizer.transform([cleaned_text])
            
        return tfidf_matrix.toarray()[0]

    def compare_with_previous_topics(self, current_segment):
        if not self.topic_history:
            return None, 0.0

        current_fingerprint = self.get_topic_fingerprint(current_segment)
        max_similarity = 0.0
        best_match_idx = None

        for idx, (topic_fingerprint, _, _) in enumerate(self.topic_history):
            if len(topic_fingerprint) != len(current_fingerprint):
                continue
                
            similarity = cosine_similarity([topic_fingerprint], [current_fingerprint])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_idx = idx

        return best_match_idx, max_similarity
    
    def extract_keywords(self, sentences, top_n=3):
        if isinstance(sentences, list):
            text = ' '.join(sentences)
        else:
            text = sentences
            
        keywords = self.keyword_extractor.extract_keywords(
            text, 
            keyphrase_ngram_range=(1, 1), 
            stop_words='english',
            top_n=top_n
        )
        
        sorted_keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in sorted_keywords[:top_n]]

    def segment_transcript(self, text):
        cleaned_sentences, original_sentences = self.preprocess_text(text)
        similarity_matrix = self.calculate_similarity_matrix(cleaned_sentences)
        initial_boundaries = self.detect_topic_boundaries(similarity_matrix)
        
        final_segments = []
        topic_mappings = []
        current_topic_id = 0
        
        start_idx = 0
        for boundary in initial_boundaries + [len(original_sentences)]:
            if boundary - start_idx < self.min_segment_size:
                continue
                
            current_segment = original_sentences[start_idx:boundary]
            matching_topic_idx, similarity = self.compare_with_previous_topics(current_segment)
            
            if matching_topic_idx is not None and similarity > self.topic_similarity_threshold:
                topic_id = matching_topic_idx
                old_fingerprint, topic_name, segments = self.topic_history[matching_topic_idx]
                segments.append(current_segment)
                new_fingerprint = self.get_topic_fingerprint([sent for seg in segments for sent in seg])
                self.topic_history[matching_topic_idx] = (new_fingerprint, topic_name, segments)
            else:
                topic_id = current_topic_id
                current_topic_id += 1
                keywords = self.extract_keywords(current_segment)
                topic_name = f"Topic {topic_id + 1}: {', '.join(keywords[:2])}"
                topic_fingerprint = self.get_topic_fingerprint(current_segment)
                self.topic_history.append((topic_fingerprint, topic_name, [current_segment]))
            
            final_segments.append(current_segment)
            topic_mappings.append(topic_id)
            start_idx = boundary
            
        return final_segments, topic_mappings, self.topic_history

    def calculate_similarity_matrix(self, sentences):
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        return cosine_similarity(tfidf_matrix)

    def detect_topic_boundaries(self, similarity_matrix):
        boundaries = []
        n_sentences = len(similarity_matrix)
        
        for i in range(self.window_size, n_sentences - self.window_size):
            prev_window = similarity_matrix[i-self.window_size:i, i-self.window_size:i]
            prev_similarity = np.mean(prev_window)
            
            next_window = similarity_matrix[i:i+self.window_size, i:i+self.window_size]
            next_similarity = np.mean(next_window)
            
            cross_window = similarity_matrix[i-self.window_size:i, i:i+self.window_size]
            cross_similarity = np.mean(cross_window)
            
            if (cross_similarity < self.similarity_threshold and
                cross_similarity < prev_similarity * 0.8 and
                cross_similarity < next_similarity * 0.8 and
                (len(boundaries) == 0 or i - boundaries[-1] >= self.window_size)):
                boundaries.append(i)
        
        return boundaries
    
if __name__ == "__main__":
    transcript = """
        Alice: Let's discuss the bug fixes for the mobile app.
    Bob: We've identified three critical issues in the payment flow.
    Charlie: The checkout process keeps crashing on Android devices.
    
    David: Sorry to interrupt, but we need to address the server outage from yesterday.
    Alice: Yes, what caused the downtime?
    Charlie: It was a database connection issue.
    David: We've implemented a fix, but we need to improve monitoring.
    
    Bob: Going back to the mobile app bugs - we need to prioritize the Android fix.
    Alice: Agreed, it's affecting too many users.
    Charlie: I can have it fixed by tomorrow.
    
    David: About the server monitoring - I propose we implement automated alerts.
    Alice: Good idea, what metrics should we track?
    David: Response time, error rates, and CPU usage.
    
    Bob: One last thing about the mobile app - we should also fix the payment flow.
    Charlie: I'll tackle that right after the Android bug.
    
    David: Quick update on the server - monitoring is now set up.
    Alice: Perfect, keep an eye on those metrics.
    """
    
    segmenter = TopicSegmenter(
        window_size=4,
        similarity_threshold=0.25,
        context_size=2,
        min_segment_size=3,
        topic_similarity_threshold=0.35
    )
    
    segments, topic_mappings, topic_history = segmenter.segment_transcript(transcript)
    
    print("Topic Segmentation Analysis:\n")
    for i, (segment, topic_id) in enumerate(zip(segments, topic_mappings)):
        print(f"Segment {i+1} (Part of {topic_history[topic_id][1]}):")
        print("-" * 50)
        print("\n".join(segment))
        print("-" * 50 + "\n")