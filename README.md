# Flowify

## 🎯 Overview
Flowify is an intelligent web application that transforms video content into intuitive, visual flow charts. By leveraging advanced NLP techniques and clustering algorithms, Flowify automatically organizes video content into coherent topics and subtopics.

## ✨ Features
- **Video Processing**: Upload and process videos of any length
- **Automatic Transcription**: Convert speech to text with high accuracy
- **Smart Topic Clustering**: 
  - Vector embeddings of transcript segments
  - Similarity matrix generation using cosine similarity
  - Dynamic topic identification through diagonal matrix sliding
- **Visual Flow Charts**: Generate clear, hierarchical visualizations of content structure

## 🛠 Technical Architecture
1. **Frontend**: Web interface for video upload and flow chart visualization
2. **Backend Processing Pipeline**:
   - Video transcription model (ran locally)
   - Text segmentation
   - Vector embedding generation
   - Clustering algorithm
   - Flow chart generation

## 🚀 Getting Started
`pip install -r requirements.txt`

`python3 app.py`

## 🤝 Contributing
This project was created during **TartanHacks**. Feel free to contribute!

## 👥 Team
**Team Hackintosh**: [Aryan Daga](https://github.com/aryand2006), [Lakshya Gera](https://github.com/lakrage), [Samatva Kasat](https://github.com/samkas125), [Yuvvan Talreja](https://github.com/yuvvantalreja)