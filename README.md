🌿 Plant AI: Smart Plant Disease Diagnosis
An intelligent application for real-time plant disease recognition. This project uses a Streamlit frontend to interact with a powerful FastAPI backend, which leverages state-of-the-art models from Hugging Face to provide instant and accurate diagnoses from leaf images. 🧑‍🌾

Live Demo
You can try the live application here:

https://plantdiagnosis.streamlit.app/

About The Project
Plant AI is designed to make advanced plant disease diagnosis accessible to everyone. The frontend allows for easy image uploads, while the backend API, deployed on Render, handles the heavy lifting of image processing and model inference. By using pre-trained vision transformers from the Hugging Face Hub, the application delivers high-accuracy predictions quickly.

Core Features
Real-time Recognition: Classifies plant diseases using powerful models from Hugging Face.

Decoupled Architecture: A responsive Streamlit UI communicates with a robust FastAPI backend.

Scalable Deployment: The backend is deployed on Render for reliable, scalable performance.

Simple Interface: A clean and straightforward user interface for uploading images and viewing results.

Tech Stack
This project is built with a modern, decoupled technology stack:

Frontend:

Streamlit

Backend:

Python

FastAPI (for the REST API)

Hugging Face Transformers (for the core AI model)

Image Processing:

Pillow

Deployment:

Render (for the backend API)

Streamlit Community Cloud (for the frontend)

Model & Dataset
Model
This project utilizes a pre-trained Vision Transformer (ViT) model from the Hugging Face Hub. The model has been fine-tuned on the PlantVillage Dataset to specialize in recognizing 38 different plant disease categories with high accuracy.

Dataset
The PlantVillage Dataset contains over 54,000 images of healthy and diseased plant leaves, providing a robust foundation for training a reliable classification model.

License
Distributed under the MIT License. See LICENSE for more information.
