# Air Reader : Gesture-Controlled PDF Viewer


Hands-free PDF navigation using:
- Hand gesture recognition (MediaPipe)
- KNN classifier (98.2% accuracy)
- PDF.js integration

## Features
- Zoom in/out with pinch gestures
- Real-time hand tracking
- Cross-platform (Windows/macOS/Linux)

## Installation
```bash
git clone https://github.com/ajydeep/Air-Reader.git
cd Air-Reader
pip install -r requirements.txt
```

## Usage
1. Run data collection:
   ```bash
   python collect_data.py
   ```
2. Train the model:
   ```bash
   python train_model.py
   ```
3. Start the viewer:
   ```bash
   python main.py
   ```

## Project Structure
```
├── pre/                  
│   ├── collect_data.py
│   ├── best_gesture_model.pkl
│   └── train_model.py
├── data/                 
│   └── gesture_data.csv
├── main.py          
├── viewer.html
└── README.md
```

## License
MIT