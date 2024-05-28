import * as React from 'react';
import { auth, googleProvider } from "../config/firebase";
import {
  createUserWithEmailAndPassword,
  signInWithPopup,
  signOut,
} from "firebase/auth";
import { useState } from "react";



export const Auth = () => {
  const [email, setEmail] = useState("");


  const signIn = async () => {
    try {
      await createUserWithEmailAndPassword(auth, email, 'bububu');
      
    } catch (err) {
      console.error(err);
    }
  };

  const signInWithGoogle = async () => {
    try {
      await signInWithPopup(auth, googleProvider);
    } catch (err) {
      console.error(err);
    }
  };

  const logout = async () => {
    try {
      await signOut(auth);
    } catch (err) {
      console.error(err);
    }
  };

  

  return (
    <div className="auth">

      <input
        placeholder="Електронна пошта"
        onChange={(e) => setEmail(e.target.value)}
      /> 
    

      <button className="b1"
      onClick={signIn}> 
        Ввійти
      </button> 
      <button className="b2"
      onClick={signInWithGoogle}
      > 
        Ввійти за допомогою Google
        </button> 
      <br/>
      <button 
      className="b3"
      onClick={logout}> Вийти </button>
    </div>
  );
};

