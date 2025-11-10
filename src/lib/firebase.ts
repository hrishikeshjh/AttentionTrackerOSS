import { initializeApp, getApps, getApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';

const firebaseConfig = {
  apiKey: "AIzaSyA7CxVTz00XIAqViDmbAr92TDYiTotGda4",
  authDomain: "studio-8196614962-36180.firebaseapp.com",
  projectId: "studio-8196614962-36180",
  storageBucket: "studio-8196614962-36180.firebasestorage.app",
  messagingSenderId: "240470473825",
  appId: "1:240470473825:web:66da9a692dfd810720fe20"
};

// Initialize Firebase
const app = !getApps().length ? initializeApp(firebaseConfig) : getApp();
const auth = getAuth(app);
const db = getFirestore(app);

export { app, auth, db };
