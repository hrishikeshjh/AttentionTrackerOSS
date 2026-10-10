> **Status note (original brief, not a description of the current code).** This is the initial product brief and is kept for history. What differs in the code: face landmarks come from face-api.js (68 points), not MediaPipe FaceMesh (MediaPipe Face Landmarker is used only in the opt-in DACE mode); there is no TensorFlow/Keras `attention_model.h5`: scoring is a weighted rule engine plus an optional ONNX temporal classifier; Firestore logging is **not implemented** (no session data is written to Firestore; Firebase is used for Auth); the data point interval is 1 s, not 5 s; tab-hidden detection exists (`visibilitychange` in `webcam-panel.tsx`) but its 3 s threshold and score penalty were not re-verified. See `v1.0/README.md` and `ARCHITECTURE.md`.

# **App Name**: CognEn: Cognitive Engagement Analyzer

## Core Features:

- User Authentication: Firebase email/password and Google Sign-in to authenticate users and store user info under /users/{uid} in Firestore.
- Facial Landmark & Attention Tracking: Capture webcam feed, use Mediapipe FaceMesh to extract EAR, MAR, and head pose, feed features into a TensorFlow/Keras CNN classifier (attention_model.h5), and output Attentive, Distracted, or Multitasking states.
- Attention Score Computation: Based on facial analysis calculate continuous attention score (0-100) influenced by blink rate, head pose and eye aspect ratio.
- Multitasking Detection: Use JavaScript to detect browser tab inactivity. Log multitasking events if the tab is hidden for more than 3 seconds and reduce the attention score.
- Data Logging: Log timestamp, attention score, multitask flag, and head pose every 5 seconds to Firestore under /users/{uid}/sessions/{session_id}.
- Real-time Dashboard Visualization: Display a live line chart of attention score vs time, a pie chart of attention states, and a counter of multitasking events. Show summary card with average attention, multitask events, engagement level, and duration.
- Session Report Generation: Tool that utilizes the data collected to generate summary report. Export the session summary as a PDF report.

## Style Guidelines:

- Primary color: Neon cyan (#00FFFF) for a vibrant and engaging feel.
- Background color: Dark background (#0D1117) to provide contrast and reduce eye strain.
- Accent color: Green (#00FF99) for highlights and interactive elements.
- Body and headline font: 'Inter', a grotesque-style sans-serif, which offers a modern and neutral look suitable for both headlines and body text.
- Use clear, minimalist icons to represent different attention states and actions.
- Divide the UI into a left panel for live webcam feed and controls, and a right panel for visualizations and session summary.
- Subtle animations to indicate changes in attention score and multitasking events.