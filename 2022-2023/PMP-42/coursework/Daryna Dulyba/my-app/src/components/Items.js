import React from 'react'
import {Item} from './Item'

export const Items = ({items,addItem,setFullItem}) =>{

        return (
            <main>
                {items.map(el => (
                    <Item  key={el.id} item={el} addItem={addItem} setFullItem={setFullItem}/>
                ))}
            </main>
        )
 
}

