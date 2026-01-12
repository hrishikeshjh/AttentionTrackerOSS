# CognEn Chrome Extension

A Chrome extension for real-time cognitive engagement tracking using facial analysis.

## Features

- **Real-time Attention Monitoring**: Track your attention score while browsing
- **PERCLOS Analysis**: Percentage of Eye Closure tracking for fatigue detection
- **Blink Rate Monitoring**: Track blink patterns for engagement insights
- **Fatigue Detection**: Get alerts when fatigue is detected
- **Tab Activity Tracking**: Monitor multitasking behavior
- **Session Statistics**: View average scores and session summaries

## Installation

### Development Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right corner
3. Click "Load unpacked"
4. Select the `extension` folder from this project
5. The CognEn icon should appear in your toolbar

### Required Permissions

- **Camera**: For facial landmark detection
- **Storage**: To save session data and settings
- **Notifications**: For attention and break reminders
- **Active Tab**: To detect tab switching (multitasking)

## Usage

1. Click the CognEn icon in your Chrome toolbar
2. Click "Start Camera" to enable webcam access
3. Click "Start Session" to begin tracking your attention
4. The badge on the extension icon will show your current score
5. View detailed metrics in the popup

## Score Interpretation

- **70-100 (Green)**: Attentive - You're focused and engaged
- **40-69 (Orange)**: Distracted - You may be losing focus
- **0-39 (Red)**: Multitasking - Low engagement detected

## Metrics Explained

- **Attention Score**: Overall engagement score (0-100)
- **PERCLOS**: Percentage of time eyes are closed (lower is better)
- **Blink Rate**: Blinks per minute (15-20 is normal)
- **Fatigue Index**: Composite fatigue score (0-100, lower is better)

## Privacy

- All processing happens locally in your browser
- No video data is sent to any server
- Session data is stored locally in Chrome's storage

## Troubleshooting

**Camera not working?**
- Check that you've granted camera permissions
- Try refreshing the page or restarting Chrome

**Models not loading?**
- Ensure you have an internet connection (models are loaded from CDN)
- Try disabling other extensions that might block scripts

**Low scores despite focusing?**
- Ensure good lighting on your face
- Position your face clearly in the camera frame
- Check that the webcam quality is sufficient

## Building for Production

To package the extension for distribution:

1. Remove any development files
2. Zip the `extension` folder
3. Upload to Chrome Web Store (requires developer account)

## Technical Details

- Uses face-api.js for face detection and landmark extraction
- Implements PERCLOS (Percentage of Eye Closure) for drowsiness detection
- Eye Aspect Ratio (EAR) for blink detection
- Head pose estimation for engagement tracking
- Exponential Moving Average for score smoothing

## License

Part of the CognEn project. See main repository for license details.
