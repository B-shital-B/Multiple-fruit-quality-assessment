Multiple Fruit Quality Assessment System
A real-time, edge-deployed system that identifies the type of a fruit (apple, banana, mango, orange) and grades its quality (fresh or rotten) using deep learning and computer vision on a Raspberry Pi 4B.

Final Year B.E. Project (Electronics & Telecommunication Engineering), 2024-25 PES's Modern College of Engineering, Pune - Savitribai Phule Pune University

📺 Demo video:https://www.youtube.com/watch?v=DMkw4FQuWWo&t=2s

📌 Overview
Manual fruit inspection is slow, subjective and error-prone. This project automates it with two lightweight MobileNetV2 models (transfer learning, ImageNet pre-trained) that are converted to TensorFlow Lite and run directly on a low-cost Raspberry Pi, with no cloud or GPU needed.

Task	Classes	Model
Fruit classification	Apple, Banana, Mango, Orange	MobileNetV2 (softmax, 4 classes)
Quality assessment	Fresh, Rotten	MobileNetV2 (softmax, 2 classes)
The novelty of the work is its dual focus: high accuracy and efficiency on resource-constrained edge hardware.

✨ Features
Real-time image capture from a USB camera
Fruit type classification and fresh/rotten grading with confidence scores
On-device inference with TensorFlow Lite (low latency, portable)
Annotated result images saved locally
Result logging to CSV (and Google Sheets)
Compact enclosure design for a portable setup
🧠 System Pipeline
Camera → Image capture → Preprocessing (resize 224×224, normalize)
      → MobileNetV2 (fruit model + quality model, TFLite)
      → Prediction + confidence → Annotated output → CSV / Sheets log
Model architecture (both models): MobileNetV2 base (frozen, ImageNet weights) → Global Average Pooling (1280-d) → Dense (128, ReLU) → Softmax output layer.

🗂️ Dataset
Based on the FruitVision dataset, a custom set of 3,200 labeled images
4 fruits × 2 quality states (fresh / rotten)
Every image annotated with both fruit type and quality
Resized to 224 × 224, normalized (÷255), augmented (rotation, flipping)
Labels managed through CSV and JSON mappings
Split: 70% train / 15% validation / 15% test
The full dataset is not included in this repository. See data/README.md for how to obtain and arrange it. A few sample images are in data/sample_images/.

📊 Results
Model	Format	Epochs	Test set	Result
Quality (fresh / rotten)	TFLite	30	480 images (240 per class)	100% accuracy; precision, recall and F1 = 1.00 for both classes
Fruit classification	-	20	-	~98.95% accuracy (Banana F1 = 1.00, Mango F1 ≈ 0.996)
Full classification report: results/quality_tflite_classification_report_30epochs.csv

Quality model (TFLite, epoch 30)
Confusion matrix

Show Image

Precision-Recall curve

Show Image

Metrics vs. confidence threshold

F1 vs. Confidence	Precision vs. Confidence	Recall vs. Confidence
Show Image	Show Image	Show Image
🔧 Hardware
Component	Specification
Raspberry Pi 4 Model B	Quad-core Cortex-A72 @ 1.5 GHz, 4 GB RAM
Camera	1/4" CMOS 640×480 USB camera (collapsible cable)
Storage	32 GB microSD (Class 10)
Power	5 V DC supply (USB-C)
Enclosure	Acrylic / ABS box-type casing
Total cost: ≈ ₹7,400 - see hardware/bill_of_materials.md.

Software Stack
Python 3.11 (Raspberry Pi OS 64-bit)
TensorFlow / Keras (training) and TensorFlow Lite (inference)
OpenCV (capture and preprocessing)
NumPy, Matplotlib, scikit-learn (evaluation and plots)
Getting Started
1. Clone the repository
bash
git clone https://github.com/B-shital-B/multiple-fruit-quality-assessment.git
cd multiple-fruit-quality-assessment
2. Install dependencies
bash
pip install -r requirements.txt
3. Train the models (optional, on PC / Google Colab)
bash
python src/train_fruit_model.py
python src/train_quality_model.py
python src/convert_to_tflite.py
Or open notebooks/train_mobilenetv2.ipynb in Google Colab.

4. Evaluate
bash
python src/evaluate.py
5. Run real-time inference on Raspberry Pi
bash
python src/raspberry_pi/realtime_inference.py
Pre-trained TFLite models are provided in models/.

⚠️ Adjust the file names above if your scripts are named differently.

📁 Repository Structure
├── README.md
├── requirements.txt
├── notebooks/        # Training notebook
├── src/              # Training, evaluation, conversion, Pi inference code
├── models/           # .tflite models
├── data/             # Dataset instructions, labels, sample images
├── results/          # Curves, confusion matrix, output samples
└── hardware/         # Block diagram, photos, BOM
 Future Scope
Multi-fruit detection in one frame using YOLOv8 / SSD / EfficientDet
Mobile app (Android Studio / Flutter) with on-device TFLite
Cloud logging and dashboards (Firebase / AWS)
Model optimization: quantization (int8), pruning, distillation
Multilingual voice feedback (Marathi, Hindi, English)
Support for more fruit types and ripeness stages
 Applications
Food processing and packaging units · Retail and supermarkets · Smart farming and post-harvest sorting · Supply chain quality control

Team
Shital Bhadre - @B-shital-B
Ibtesam Pathan
Sahil Kangane
Project Guide: Prof. Ms. M. M. Deshpande HOD (E&TC): Prof. Dr. Mrs. R. S. Kamthe

 Documentation
Full project report: Project Report





