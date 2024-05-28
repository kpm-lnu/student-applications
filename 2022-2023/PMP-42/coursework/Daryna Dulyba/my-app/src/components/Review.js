import { useEffect, useState } from "react";
import { db, auth } from "../config/firebase";
import {
  getDocs,
  collection,
  addDoc,
  deleteDoc,
  doc,
  serverTimestamp,
} from "firebase/firestore";
import {
  createUserWithEmailAndPassword,
  
} from "firebase/auth";
import { FaTrash } from "react-icons/fa";
import { Grid } from "@mui/material";


export default function Review() {

  const [review, setReview] = useState([]);
  const [newReview, setNewReview] = useState("");
  const [email, setEmail] = useState("");

  const reviewsCollectionRef = collection(db, "reviews");

  const getReview = async () => {
    try {
      const data = await getDocs(reviewsCollectionRef);
      const filteredData = data.docs.map((doc) => ({
        ...doc.data(),
        id: doc.id,
      }));
      setReview(filteredData);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    getReview();
  }, []);



  const onSubmitReview = async () => {
    try {
      await createUserWithEmailAndPassword(auth, email, '12345678');
      await addDoc(reviewsCollectionRef, {
        content: newReview,
        user: auth.currentUser.email,
        createdAt: serverTimestamp(),
        userId: auth?.currentUser?.uid,
      });
      getReview();
    } catch (err) {
      console.error(err);
    }
  };



  const deleteReview = async (id) => {
    const reviewDoc = doc(db, "reviews", id);
    await deleteDoc(reviewDoc);
    getReview();
  };

  return (
    <div className="review">
      <div 
      style={{alignItems:'center', 
      justifyContent:'center', 
      width:'100%', 
      display:'flex', 
      marginBottom:'15px', 
      fontSize:'23px', 
      fontWeight:'600'}}>
        Відгуки
        </div>

      <Grid container style={{marginTop:'40px'}}>
      <Grid xs={6}>
      <div className="auth"> 

      <input
        placeholder="Електронну пошта"
        onChange={(e) => setEmail(e.target.value)}
      /> 
      </div>
        <textarea
        placeholder="Напишіть свій відгук.."
        onChange={(e) => setNewReview(e.target.value)}
        /> <br /> <br/>
        <button onClick={onSubmitReview} disabled={!email || !newReview}> Надіслати відгук </button> <br/> <br/>
        <span style={{fontSize:'14px'}}>*Для того, щоб залишити відгук введіть електрону пошту.</span> 
      </Grid>
      
      <Grid
      className="main-content"
      xs={6}>
        {review.map((review) => (
          <div className="content">
            <b>{review.user }</b> 
            <br/>
            {review.content}

            <FaTrash 
            className="delete-icon" 
            onClick={() => deleteReview(review.id)} />
          </div>
        ))}
      </Grid>
      </Grid>  
    </div>
  );
}
