# CognEn Chrome Extension

> **Scoring parity.** `lib/ml-core.js` is a plain-JS port of the web app's `src/lib/scoring/` package and the legacy `MLEngine` path of `src/lib/ml-engine.ts`: same EAR/MAR, head pose, gaze, blink detector (50-500 ms closures, no extra debounce), PERCLOS, fatigue (PERCLOS x2 capped at 40, EAR trend over 10 samples x100), fidget (x5), stability, `PrimaryFaceSelector` hysteresis, aggregator outputs (incl. `eyebrowRaise`, `marVariance`, `poseVariance`), EMA alpha 0.15 and thresholds 70/40. The extension has no object detector or hand tracker, so both modes score with `FEATURE_WEIGHTS_LEGACY` (what the web engine uses when `detectedObjects` is undefined). The context features (`contextAwarePitch`, `objectContext`, `handObjectProximity`, `temporalPattern`) are implemented with the web semantics but are not weighted. `src/lib/__tests__/extension-equivalence.test.ts` loads `ml-core.js` in node and checks it against the TS modules on synthetic landmark sequences. The pitch-baseline defect described in `v1.0/README.md` §24 applies here too, because the formulas are the same.

A Chrome extension for real-time cognitive engagement tracking using facial analysis.

## Features

- **Real-time Attention Monitoring**: Track your attention score while browsing
- **PERCLOS Analysis**: Percentage of Eye Closure tracking for fatigue detection
- **Blink Rate Monitoring**: Track blink patterns for engagement insights
- **Fatigue Detection**: Get alerts when fatigue is detected
- **Session Statistics**: View average scores and session summaries

## Installation

### Development Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right corner
3. Click "Load unpacked"
4. Select the `extension` folder from this project
5. The CognEn icon should appear in your toolbar

### Required Permissions

- **Camera** (requested by the offscreen document via `getUserMedia`): facial landmark detection in standalone mode.
- **storage**: settings, last detection, and the session record (`chrome.storage.session`, falling back to `chrome.storage.local`).
- **offscreen**: standalone-mode webcam capture in a hidden document.
- **scripting** + **activeTab** + host permissions for `meet.google.com`, `*.zoom.us`, `teams.microsoft.com`: injecting the in-page analyser on call pages.
- **alarms**: break reminders (survive service-worker restarts).
- **notifications**: low-attention alerts and break reminders (both controlled by the popup settings).
- `web_accessible_resources` exposes only `lib/*`, and only to the three call hosts above (it was `<all_urls>`).

### Cadence and performance

Both modes analyse frames on a fixed 200 ms `setTimeout` tick (5 FPS; `requestAnimationFrame` is not used because rAF is paused in hidden/offscreen documents) and predict once per second, like the web app (`DATA_POINT_INTERVAL` = 1000 ms). This gives the 0.15 EMA the same time constant in the extension and the web app. One detection per second goes to the background, which owns the badge, the session statistics and the alerts.

### Settings and alerts

- **Alert threshold** (default 40): if the score stays below it for 30 s, a notification fires (at most one every 5 min).
- **Break reminder** (default 45 min): a `chrome.alarms` alarm that fires a notification while a session is active.
- **Notifications** toggle turns both off. **Sound alerts** decides whether the notification is silent.

### Meet / Zoom / Teams mode: messaging channel

The analyser has to run in the page's MAIN world (it reads the page's `<video>` elements), which it shares with the page's own scripts. To limit what those scripts can do:

1. `background.js` generates a random 256-bit key for each session. It sends the key through `chrome.scripting.executeScript` `args` to `bridge.js` (ISOLATED world) and to `content.js` (MAIN world). The key never goes into the DOM, storage or `postMessage`.
2. `content.js` signs every message (`seq|kind|payload`) with HMAC-SHA-256 and posts it with `window.postMessage`.
3. `bridge.js` checks the MAC with the isolated world's own `crypto.subtle`, rejects replays (`seq` must increase), whitelists and range-checks the payload, and relays it with `chrome.runtime.sendMessage`.
4. `background.js` accepts it only from the session tab and only with the current session id.

The old `localStorage` channel, which any page script could read or write, has been removed.

**What is still exposed:** page scripts can read the posted messages (scores, blink rate, fatigue) and the widget, and can call `window.__cognen.content.stop()` or break the injected globals. A page that patched `crypto.subtle` or `postMessage` before injection could capture the key. Without that kind of tampering, a page cannot forge or replay accepted messages.

All injected scripts are wrapped in IIFEs and attach to a single `window.__cognen` namespace with sentinels, so a second START reuses them and does not throw on redeclared bindings.

### Which video is analysed

The self-view tile is picked by ranking the playing `<video>` elements:

1. A `srcObject` with a live local-capture track (`getSettings().deviceId` set; remote WebRTC tracks have none).
2. A horizontally mirrored element or close ancestor (negative x scale, e.g. `scaleX(-1)`). All three apps mirror only the self-view.
3. If neither applies, the largest visible area.

The widget's **Switch video** button cycles through the playing videos and resets the scoring history.

## Usage

1. Click the CognEn icon in your Chrome toolbar
2. Click "Start Session" (standalone uses your webcam; on a Meet/Zoom/Teams tab the in-page video is analysed)
3. The badge on the extension icon will show your current score
4. View detailed metrics in the popup

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
- Session data is stored locally in Chrome's storage (session summary in `lastSession`)

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
