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

        for idx, (topic_fingerprint, _, _, _) in enumerate(self.topic_history):
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

    def find_closest_timestamp(self, sentence, transcript):
        """
        Find the timestamp after the most recent timestamp before the sentence.
        """
        chunks = re.findall(r'\[(\d+)\](.*?)(?=\[\d+\]|$)', transcript, re.DOTALL)
        sentence = sentence.strip().lower()
        
        previous_timestamp = 0
        current_timestamp = 0
        
        for i, (timestamp, text) in enumerate(chunks):
            current_timestamp = int(timestamp)
            text = text.strip().lower()
            
            if sentence in text or text in sentence:
                # Try to get the timestamp after the previous timestamp
                if i > 0 and i < len(chunks):
                    return current_timestamp
                return previous_timestamp
            
            previous_timestamp = current_timestamp
                
        return current_timestamp  # Return last timestamp if sentence not found

    
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
                old_fingerprint, topic_name, segments, _ = self.topic_history[matching_topic_idx]
                segments.append(current_segment)
                new_fingerprint = self.get_topic_fingerprint([sent for seg in segments for sent in seg])
                closest_timestamp = self.find_closest_timestamp(current_segment[-1], text)
                self.topic_history.append((new_fingerprint, topic_name, segments, closest_timestamp))
            else:
                topic_id = current_topic_id
                current_topic_id += 1
                keywords = self.extract_keywords(current_segment)
                topic_name = f"Topic {topic_id + 1}: {', '.join(keywords[:2])}"
                topic_fingerprint = self.get_topic_fingerprint(current_segment)
                closest_timestamp = self.find_closest_timestamp(current_segment[-1], text)
                self.topic_history.append((topic_fingerprint, topic_name, [current_segment], closest_timestamp))
            
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
[0] Translator: Joseph Geni
Reviewer: Morton Bast There are a lot of ways
the people around us can help improve our lives We don't bump into every neighbor, so a lot of wisdom never gets passed on,. [26] though we do share the same public spaces So over the past few years,
I've tried ways to share more with my neighbors in public space, using simple tools like
stickers, stencils and chalk And these projects came
from questions I had, like:. [41] How much are my neighbors
paying for their apartments? (Laughter) How can we lend and borrow more things, without knocking on each
other's doors at a bad time? How can we share more memories
of our abandoned buildings,. [56] and gain a better understanding
of our landscape? How can we share more of our hopes
for our vacant storefronts, so our communities can reflect
our needs and dreams today? Now, I live in New Orleans, and I am in love with New Orleans. [74] My soul is always soothed
by the giant live oak trees, shading lovers, drunks and dreamers
for hundreds of years, and I trust a city that always
makes way for music I feel like every time someone sneezes, New Orleans has a parade. [90] (Laughter) The city has some of the most
beautiful architecture in the world, but it also has one of the highest amounts
of abandoned properties in America I live near this house, and I thought about how I could
make it a nicer space for my neighborhood,. [105] and I also thought about something
that changed my life forever In 2009, I lost someone I loved very much Her name was Joan,
and she was a mother to me And her death was sudden and unexpected And I thought about death a lot. [132] And  this made me feel deep
gratitude for the time I've had And  brought clarity to the things
that are meaningful to my life now But I struggle to maintain
this perspective in my daily life I feel like it's easy to get
caught up in the day-to-day, and forget what really matters to you. [160] So with help from old and new friends, I turned the side of this abandoned 
house into a giant chalkboard, and stenciled it with
a fill-in-the-blank sentence: "Before I die, I want to " So anyone walking by
can pick up a piece of chalk,. [176] reflect on their life, and share their personal
aspirations in public space I didn't know what to expect
from this experiment, but by the next day,
the wall was entirely filled out, and it kept growing. [191] And I'd like to share a few things
that people wrote on this wall "Before I die, I want
to be tried for piracy" (Laughter) "Before I die, I want to straddle
the International Dateline" "Before I die, I want
to sing for millions". [218] "Before I die, I want to plant a tree" "Before I die, I want
to live off the grid" "Before I die, I want
to hold her one more time" "Before I die, I want
to be someone's cavalry" "Before I die, I want
to be completely myself". [249] So this neglected space
became a constructive one, and people's hopes and dreams
made me laugh out loud, tear up, and they consoled me
during my own tough times It's about knowing you're not alone; it's about understanding our neighbors
in new and enlightening ways;. [267] it's about making space
for reflection and contemplation, and remembering what really matters
most to us as we grow and change I made this last year, and started receiving hundreds
of messages from passionate people who wanted to make a wall
with their community. [284] So, my civic center colleagues
and I made a tool kit, and now walls have been made
in countries around the world, including Kazakhstan, South Africa, Australia,. [299] Argentina, and beyond Together, we've shown how powerful        
our public spaces can be if we're given the opportunity
to have a voice, and share more with one another Two of the most valuable things we have. [315] are time, and our relationships
with other people In our age of increasing distractions, it's more important than ever
to find ways to maintain perspective, and remember that life
is brief and tender Death is something that we're
often discouraged to talk about,. [332] or even think about, but I've realized that preparing for death is one of the most empowering
things you can do Thinking about death clarifies your life Our shared spaces can better
reflect what matters to us,. [347] as individuals and as a community, and with more ways to share
our hopes, fears and stories, the people around us can not only
help us make better places, they can help us lead better lives Thank you. [362] (Applause) Thank you (Applause)
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
    print(topic_history)
    for i, (segment, topic_id) in enumerate(zip(segments, topic_mappings)):
        print(f"Segment {i+1} (Part of {topic_history[topic_id][1]}):")
        print(f"Topic Sentence: {segment[0]}")  # Print first sentence as topic sentence
        print(f"Closest Timestamp: {topic_history[topic_id][3]} seconds")
        print("-" * 50)
        print("\n".join(segment))
        print("-" * 50 + "\n")