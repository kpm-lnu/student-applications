import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { getFirestore } from "firebase/firestore";


const firebaseConfig = {
  apiKey: "AIzaSyAwZkOYCJIuNTl5QQuRejWrgR7-J1KGQng",
  authDomain: "diplom-59430.firebaseapp.com",
  projectId: "diplom-59430",
  storageBucket: "diplom-59430.appspot.com",
  messagingSenderId: "613870991863",
  appId: "1:613870991863:web:f1004a550fe5634026d6f2"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();

export const db = getFirestore(app);
