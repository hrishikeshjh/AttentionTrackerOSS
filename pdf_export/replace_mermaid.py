import re

with open('README-overall.md', 'r') as f:
    content = f.read()

# Replace v1.0 mermaid block
v1_mermaid = r'```mermaid\nflowchart LR\n    A\[Webcam\\n640×480 @ 5 FPS\] --> B\[face-api\.js\\nTinyFaceDetector\]\n    B --> C\[68-pt Landmarks\]\n    C --> D\[FaceMetricsAggregator\\nEAR · MAR · Gaze\\nHead Pose · Blink\\nFatigue · PERCLOS\]\n\n    A --> E\[YOLOv8-nano ONNX\\nevery 3rd frame\]\n    E --> F\[Detected Objects\\nbook · phone · laptop\]\n\n    A --> G\[MediaPipe Hands\\nevery 3rd frame\]\n    G --> H\[Hand–Object\\nProximity\]\n\n    D & F & H --> I\[Feature Fusion\\n13-dim vector\]\n    I --> J\[TemporalBehaviorClassifier\\n1D-CNN · 15-frame window\]\n    J --> K\[Context-Gate\\nOverride\]\n    K --> L\[MLEngine\\nContext-Aware Scoring\\nEMA α=0\.15\]\n    L --> M\[Attention Score 0–100\\nBehavior Label\]\n    M --> N\[Dashboard\\n\+ Firebase \+ PDF\]\n```'

# More robust regex replacement for mermaid blocks
v1_pattern = re.compile(r'### 3\.1 v1\.0 — Browser Pipeline\n\n```mermaid\n(.*?)\n```', re.DOTALL)
content = v1_pattern.sub(r'### 3.1 v1.0 — Browser Pipeline\n\n![v1.0 Browser Pipeline](images/v1_pipeline.png)', content)

v2_pattern = re.compile(r'### 3\.2 v2\.0 — Server Pipeline\n\n```mermaid\n(.*?)\n```', re.DOTALL)
content = v2_pattern.sub(r'### 3.2 v2.0 — Server Pipeline\n\n![v2.0 Server Pipeline](images/v2_pipeline.png)', content)

with open('pdf_export/README-rendered.md', 'w') as f:
    f.write(content)
