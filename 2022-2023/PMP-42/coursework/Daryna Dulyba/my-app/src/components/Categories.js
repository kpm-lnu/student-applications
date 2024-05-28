
import React from 'react'

export const Categories = ({setCategory,category})=>{

    const cathegories = [
        {
            key: 'all',
            name: 'Всі'
        },
        {
            key: 'tulips',
            name: 'Тюльпани'
        },
        {
            key: 'rose',
            name: 'Троянди'
        },
        {
            key: 'daisies',
            name: 'Ромашки'
        },
        {
            key: 'peony',
            name: 'Півонії'
        }
    ];

 
    return (
      <div className='categories'>
        {cathegories.map(el => (
            <div style={{borderColor: category===el.key && '#c18cc7', borderWidth:category===el.key && '2px' }} key={el.key} onClick={() => setCategory(el.key)}> {el.name} </div>
        ))}
      </div>
    )

};

