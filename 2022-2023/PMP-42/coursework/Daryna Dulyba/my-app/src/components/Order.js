import React from 'react'
import { FaTrash } from "react-icons/fa";
import { FaMinus } from "react-icons/fa";
import { FaPlus } from "react-icons/fa";

export const Order = ({item,removeItem,addItem}) => {

    return (
      <div className='item'>
        <img src={item.img} alt='img' />
        <h2>{item.title}</h2>
        <b>{(item.price)*(item.count)}₴</b>
        <div className='more-less-button'>
          <div onClick={()=>removeItem(item.id)}><FaMinus className='remove-item'/>
            </div> <b>{item.count}</b> <div  onClick={()=>addItem(item.id)}><FaPlus className='add-item' /></div>
          </div>
        <FaTrash className='delete-icon' onClick={()=>removeItem(item.id,true)}/>
      </div>
    )
};
