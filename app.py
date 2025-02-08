from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from topic_segmenter import TopicSegmenter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
        return jsonify({'message': 'File uploaded successfully', 'filepath': filepath})

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/analyze', methods=['POST'])
def analyze_transcript():
    if 'transcript' not in request.json:
        return jsonify({'error': 'No transcript provided'}), 400

    transcript = request.json['transcript']

    # Initialize and run topic segmentation
    segmenter = TopicSegmenter(
        window_size=4,
        similarity_threshold=0.25,
        context_size=2,
        min_segment_size=3,
        topic_similarity_threshold=0.35
    )

    try:
        segments, topic_mappings, topic_history = segmenter.segment_transcript(transcript)

        # Format results for frontend, preserving timestamps
        results = []
        for i, (segment, topic_id) in enumerate(zip(segments, topic_mappings)):
            # Preserve the original text with timestamps
            results.append({
                'segment_id': i + 1,
                'topic_name': topic_history[topic_id][1],
                'content': segment  # Keep original timestamped text
            })

        return jsonify({
            'success': True,
            'segments': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)