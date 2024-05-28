import React from 'react'
import { Order } from './Order';

export const Cart = ({items,removeItem,addItem,setPage,closeCart})=>{
    const selectedItems = items.filter((item)=> item.count);

    let summa = 0;
    items.forEach(el => { summa += Number.parseFloat((el.price)*(el.count))
    });



  return  (

        <div className='shop-cart'>
        {  
            selectedItems.length ? selectedItems.map((item) => <Order item={item} removeItem={removeItem} addItem={addItem} />) :
            (<div className='empty'>
            <h2>Корзина порожня</h2>
            </div>)
        }
        
        <p className='summa'>Сума: {summa}₴</p>
        {selectedItems.length ? (

            <span className='button-to-order' onClick={() => { setPage(4); closeCart();  }}>
                Замовити
            </span>
        ) : (
            <span className='button-to-order' onClick={() => { alert("Корзина порожня") }}>
            Замовити
            </span>
        )}
    </div>
    )
}