from flask import Flask, render_template, request, jsonify
import os
import tempfile
import shutil
from werkzeug.utils import secure_filename
from vosk import Model, KaldiRecognizer
import wave
import subprocess
import json
from topic_segmenter import TopicSegmenter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size

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

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        print("No audio/video file provided")
        return jsonify({'error': 'No audio/video file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        print("Invalid file type")
        return jsonify({'error': 'Invalid file type'}), 400

    # Save the uploaded file to a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        temp_file_path = os.path.join(temp_dir, secure_filename(file.filename))
        file.save(temp_file_path)

        # Convert video/audio to WAV format
        wav_file_path = os.path.join(temp_dir, "audio.wav")
        subprocess.run(
            ["ffmpeg", "-i", temp_file_path, "-ar", "16000", "-ac", "1", wav_file_path],
            check=True
        )

        # Transcribe using Vosk
        model = Model("vosk-model-small-en-us-0.15")
        recognizer = KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)

        with wave.open(wav_file_path, "rb") as wf:
            results = []
            current_time = 0
            next_timestamp = 0

            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    if "result" in result:
                        for word_info in result["result"]:
                            if word_info["end"] >= next_timestamp:
                                results.append(f"[{int(next_timestamp)}]")
                                next_timestamp += 10
                            results.append(word_info["word"])

            # Append the final result
            final_result = json.loads(recognizer.FinalResult())
            if "result" in final_result:
                for word_info in final_result["result"]:
                    if word_info["end"] >= next_timestamp:
                        results.append(f"[{int(next_timestamp)}]")
                        next_timestamp += 20
                    results.append(word_info["word"])

        # Combine results into a single transcript
        transcript = " ".join(results)

        return jsonify({'success': True, 'transcript': transcript})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        shutil.rmtree(temp_dir)

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

        # Format results for frontend, including timestamps
        results = []
        for i, (segment, topic_id) in enumerate(zip(segments, topic_mappings)):
            # Calculate the timestamp for the segment (10-second intervals)
            timestamp = i * 10
            results.append({
                'segment_id': i + 1,
                'topic_name': topic_history[topic_id][1],
                'content': segment,
                'timestamp': timestamp  # Include timestamp
            })

        return jsonify({
            'success': True,
            'segments': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)