# CognEn Chrome Extension

> **Scope note.** The extension is **out of scope for DACE** and keeps hand-copied legacy formulas (`lib/ml-core.js`) that diverge from the web app's `src/lib/scoring/` (EMA alpha 0.35 vs 0.15, different fatigue/blink/fidget constants, no scoring case for the context-aware features, so those contribute a constant 50). Score color bands are not uniform across extension surfaces (ml-core and the content widget use 80/50, the badge 70/40, the popup ring 75/45). The legacy pitch defect described in `v1.0/README.md` §24 applies here too. Nothing in the extension has automated tests.

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
- **Active Tab**: To detect tab switching (multitasking), and to cover ad-hoc standalone-mode camera access on any page the user explicitly invokes the extension on
- **Host permissions** (updated): scoped to `meet.google.com`, `*.zoom.us`, and `teams.microsoft.com` only — a previously-present `<all_urls>` entry was removed since it granted unnecessary access beyond what Meet-mode content-script injection actually needs. See the main [v1.0 README §23](../README.md#23-security--dependency-hygiene) for the full security/dependency hygiene pass this was part of.

### Performance

The offscreen-document detection loop is now throttled to a 200ms (5fps) budget instead of running unthrottled on every display-refresh tick (~60fps) — cutting CPU usage roughly 10× during active sessions, since only 5fps of data is ever consumed downstream anyway. Storage writes are fire-and-forget and debounced (only on a meaningful score change or a 500ms cap) rather than blocking the detection loop. See [v1.0 README §20 & §22](../README.md#20-shared-scoring-core--cross-language-consistency) for the shared scoring core and testing details.

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
