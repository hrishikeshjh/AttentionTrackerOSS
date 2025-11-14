# CognEn: Cognitive Engagement Analyzer

CognEn is a simple, user-friendly web app that helps measure how focused someone is while working or studying. It uses the webcam, basic facial cues, and in-browser activity to understand attention levels in real time.

---

## 🚀 What This App Does

### **1. Sign Up & Login**
You can log in using:
- Email & password  
- Google Sign-in  

Your information is stored safely in Firestore under your unique user ID.

---

### **2. Tracks Facial Cues to Estimate Attention**
With your permission, CognEn uses your webcam to look at basic facial movements such as:
- Eye openness (to see if you're sleepy or blinking too much)  
- Mouth movement (helps detect yawning)  
- Head direction (if you’re facing away from the screen)

These signals are fed into a trained model that identifies whether you are:
- **Attentive**
- **Distracted**
- **Multitasking**

---

### **3. Gives You an Attention Score (0–100)**
The app calculates a score based on your facial cues.  
Higher score = more focused.  
Lower score = you’re probably distracted or looking away.

---

### **4. Detects Multitasking**
If you leave the browser tab or switch to another app, the system marks it as “multitasking” and reduces your score.

---

### **5. Saves Your Activity**
Every 5 seconds, CognEn stores:
- The attention score  
- Whether you multitasked  
- Your head pose  
- A timestamp  

This goes into Firestore under your session.

---

### **6. Shows a Live Dashboard**
You’ll see:
- A live attention score graph  
- A pie chart of how much time you spent attentive/distracted  
- Number of multitasking events  
- A summary with your overall engagement level  

---

### **7. Create a Session Report**
After finishing a session, you can export a clean PDF report summarizing:
- Average attention  
- Multitasking incidents  
- Time spent  
- Overall engagement rating  

---

## 🎨 Design Choices

To make the experience pleasant and easy on the eyes:

- **Primary color:** Neon Cyan (#00FFFF)  
- **Accent color:** Green (#00FF99)  
- **Background:** Dark mode (#0D1117)  
- **Font:** Inter (modern and clean)  
- **UI Layout:**  
  - Left side → Webcam preview & controls  
  - Right side → Charts & session summary  
- **Smooth animations** to show score changes or multitasking moments  

---

## 📦 Installation & Setup (Simple Steps)

1. Clone the repository  
2. Install dependencies  
3. Configure Firebase keys  
4. Place the attention model file (`attention_model.h5`) in the correct folder  
5. Run the development server  
6. Open your browser and start a session  

---

## ❤️ Why We Built This
CognEn helps students, professionals, and learners understand how engaged they really are.  
It turns simple facial signals into meaningful insights—without complexity.

---

## 📜 License
Open-source and free to use.