import React, { useState } from 'react'
import { Categories } from './Categories'
import { Items } from './Items'
import { ShowFullItem } from './ShowFullItem'

export const Main = ({addItem, allItems}) => {    
    const [category,setCategory] = useState('all');
    const [fullItem,setFullItem] = useState(null);

    const items = allItems.filter((item)=> category === 'all' || category === item.category);

  return (
    <div>
        <div className="presentation"></div>
        <Categories setCategory={setCategory} category={category} />
        <Items items={items} addItem={addItem} setFullItem={setFullItem}/> 
        <ShowFullItem  fullItem={fullItem}  setFullItem={setFullItem}/>
        
    </div>
    
  )
}







