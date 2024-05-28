import { Grid } from '@mui/material';
import React from 'react'
import { MakeOrderItems } from './MakeOrderItems';

export const MakeOrder = ({items,removeItem,addItem,setPage})=>{
    const selectedItems = items.filter((item)=> item.count);

    let summa = 0;
    items.forEach(el => { summa += Number.parseFloat((el.price)*(el.count))
    });

    return (
        <div>
            <div style={{width:'700px', marginLeft:'auto', marginRight:'auto'}}>
            <div style={{fontSize:'23px', fontWeight:'600', marginBottom:'40px', justifyContent:'center', display:'flex'}}>
                Ваша корзина
            </div>
            <Grid container style={{marginBottom:'30px', fontSize:'18px'}}>
                <Grid xs={5}>
                Найменування
                </Grid>
                <Grid xs={2}>
                Вартість
                </Grid>
                <Grid xs={2}>
                Кількість
                </Grid>
                <Grid xs={2}>
                Сума
                </Grid >
                <Grid xs={1}>

                </Grid>
            </Grid>

            { selectedItems.map((item)=> <MakeOrderItems item={item} removeItem={removeItem} addItem={addItem} />) }

                <div style={{fontSize:'18px', fontWeight:'600', marginBottom:'40px', float:'right'}}>
                    Загальна сума: {summa}₴
                </div>          
        </div>
            <div className='button-order' onClick={()=>setPage(5)}>
                Оформити замовлення
            </div>
        </div>
  )
}
