<div align="left">
  <img src="assets/logo.png" alt="flowify" width="200"/>
</div>

## Overview
Flowify is an intelligent web application that transforms video content into intuitive, visual mind maps. By leveraging advanced NLP techniques and clustering algorithms, Flowify automatically organizes video content into coherent topics and subtopics. Flowify is purely algorithmic, and doesn't require any generative AI.

## Features
- **Video Processing**: Upload and process videos of any length
- **Automatic Transcription**: Convert speech to text with high accuracy
- **Vector embeddings** of transcript segments
- **Similarity matrix generation** using cosine similarity
- Dynamic **topic identification** through **sliding submatrix window** across diagonal
- **Visual Flow Charts**: Generate clear, hierarchical visualizations of content structure

## Technical Architecture
1. **Frontend**: Web interface for video upload and flow chart visualization
2. **Backend Processing Pipeline**:
   - Video transcription model (ran client side)
   - Text segmentation    
   - Vector embedding generation (nltk TfIdVectorizer)
   - Clustering algorithm (sliding submatrix window)
   - Flow chart generation (JSMind)

## Getting Started
`pip install -r requirements.txt`

`python3 app.py`

## Contributing
This project was created during **TartanHacks 2025**. Feel free to contribute!

## Team
**Team Hackintosh**: [Aryan Daga](https://github.com/aryand2006), [Lakshya Gera](https://github.com/lakrage), [Samatva Kasat](https://github.com/samkas125), [Yuvvan Talreja](https://github.com/yuvvantalreja)