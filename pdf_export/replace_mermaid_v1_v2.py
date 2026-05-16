import re
import os

def replace_mermaid(file_path, output_path, replacements):
    with open(file_path, 'r') as f:
        content = f.read()
    
    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open(output_path, 'w') as f:
        f.write(content)

# V1 replacements
v1_replacements = [
    (r'```mermaid\nflowchart LR\n    A\["Webcam<br/>640×480 @ 5 FPS"\].*?```', r'![v1.0 Overview](images/overview.png)'),
    (r'### 3\.1 High-Level Data Flow\n\n```mermaid\nflowchart TD\n    subgraph Input\["Input Layer"\].*?```', r'### 3.1 High-Level Data Flow\n\n![v1.0 Data Flow](images/dataflow.png)'),
    (r'### 3\.2 Sequence Diagram — Single Frame Processing\n\n```mermaid\nsequenceDiagram\n    participant W as Webcam.*?```', r'### 3.2 Sequence Diagram\n\n![v1.0 Sequence Diagram](images/sequence.png)'),
    (r'### 11\. Chrome Extension\n\n\*\*Manifest V3\*\* extension with dual-mode operation:\n\n```mermaid\nflowchart LR\n    POP\["popup\.js<br/>\(UI\)"\].*?```', r'### 11. Chrome Extension\n\n**Manifest V3** extension with dual-mode operation:\n\n![v1.0 Extension](images/extension.png)')
]

# V2 replacements
v2_replacements = [
    (r'### 2\.1 Full Pipeline\n\n```mermaid\nflowchart TD\n    SRC\["Video Source<br/>RTSP · File · Camera"\].*?```', r'### 2.1 Full Pipeline\n\n![v2.0 Pipeline](images/pipeline.png)'),
    (r'### 2\.2 Sequence Diagram\n\n```mermaid\nsequenceDiagram\n    participant S as FrameSource.*?```', r'### 2.2 Sequence Diagram\n\n![v2.0 Sequence Diagram](images/sequence.png)'),
    (r'### 6\.1 Model Architecture\n\n```mermaid\nflowchart LR\n    IN\["Input<br/>\(B, 13, 15\)"\].*?```', r'### 6.1 Model Architecture\n\n![v2.0 CNN Architecture](images/cnn.png)')
]

replace_mermaid('v1.0/README.md', 'pdf_export/v1/CognEn-v1-Documentation.md', v1_replacements)
replace_mermaid('v2.0/README.md', 'pdf_export/v2/CognEn-v2-Documentation.md', v2_replacements)
