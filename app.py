from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from topic_segmenter import TopicSegmenter
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # The transcript would be generated here using the Whisper code
            # For now, we'll use a placeholder transcript
            transcript = request.form.get('transcript', '')
            
            # Process transcript with topic segmenter
            segmenter = TopicSegmenter(
                window_size=4,
                similarity_threshold=0.25,
                context_size=2,
                min_segment_size=3,
                topic_similarity_threshold=0.35
            )
            
            segments, topic_mappings, topic_history = segmenter.segment_transcript(transcript)
            
            # Format results
            results = []
            for i, (segment, topic_id) in enumerate(zip(segments, topic_mappings)):
                topic_info = topic_history[topic_id]
                results.append({
                    'segment_id': i + 1,
                    'topic_name': topic_info[1],
                    'content': '\n'.join(segment)
                })
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)