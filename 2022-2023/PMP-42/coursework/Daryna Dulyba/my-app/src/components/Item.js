import React from 'react'

export const Item = ({item,addItem,setFullItem})=> {

  const handleClick = (e) => {
    e.stopPropagation();
    addItem(item.id);
  }

    return (
      <div className='item' onClick={ ()=>setFullItem(item)}>
        <img src={ item.img}  alt='flower'/>
        <h2>{item.title}</h2>
        <p>{item.desc}</p>
        <b>{item.price}₴</b>
        <div className='add-to-cart' onClick={handleClick}>+</div>
      </div>
    )

};
