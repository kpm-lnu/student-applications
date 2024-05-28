import React, { useState } from 'react'
import { FaShoppingCart } from "react-icons/fa";
import { Cart } from './Cart';

export const Header = ({setPage, items,removeItem,addItem })=>  {
    const [cartOpen, setCartOpen] = useState(false)


  return (
    <header>
        <div>
            <span onClick={()=>{setPage(0)}} className='logo'>FlowersLand</span>
            <ul className='nav'>      
                <li onClick={()=>{setPage(1)}}>Про нас</li>
                <li onClick={()=>{setPage(2)}}>Контакти</li>
                <li onClick={()=>{setPage(3)}}>Додати відгук</li>
            </ul>

             <FaShoppingCart onClick={() => setCartOpen(!cartOpen)} className={`shop-cart-button ${cartOpen && 'active'}`}/>
            {cartOpen && (
                <Cart items={items} removeItem={removeItem} addItem={addItem} setPage={setPage} closeCart={()=>{setCartOpen(false)}}/>
            )}
        </div>
    </header>
  )
}
