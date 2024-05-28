import React from 'react'
import { MdClose } from "react-icons/md";

export const ShowFullItem = ({fullItem,setFullItem})=> {

    return ( fullItem ?  <div className='full-item'>
    <div>   
        <MdClose className='button-close' onClick={()=>setFullItem(null)} />
        <img src={fullItem.img} alt='flower'/>
        <h2>{fullItem.title}</h2>
        <p>{fullItem.desc}</p>
        <b>{fullItem.price}₴</b>
        <div className='add-to-cart' 
        >+</div>
    </div>
  </div> : null

     
    )
  };


