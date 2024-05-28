import { Grid } from '@mui/material';
import React from 'react'
import { FaTrash } from "react-icons/fa";
import { FaMinus } from "react-icons/fa";
import { FaPlus } from "react-icons/fa";


export const MakeOrderItems = ({item,removeItem,addItem}) => {
    

    return (
        <div className='make-order-items'>

            <Grid container style={{marginTop:'20px', marginBottom:'30px', border:'1px solid #747474', padding:'13px', borderRadius:'15px'}}>
                <Grid xs={5} style={{display:'flex', alignItems: 'center'}}>
                    <img style={{width:'100px', height:'100px'}} src={item.img} alt='img' /> 
                    <div style={{marginLeft:'10px'}}>
                        <div style={{fontWeight:'bold'}}>{item.title}</div>
                        <div style={{fontSize:'14px'}}>{item.desc}</div>
                    </div>
                </Grid>
                <Grid xs={2} style={{marginTop:'20px'}}>
                    <div style={{fontWeight:'bold', alignItems:'center'}}>{(item.price)}₴</div>
                </Grid>
                <Grid xs={2} style={{marginTop:'15px'}} >
                <div className='more-less-button'>
                    <div onClick={()=>removeItem(item.id)}><FaMinus className='remove-item'/>
                    </div> <b>{item.count}</b> <div  onClick={()=>addItem(item.id)}><FaPlus className='add-item' /></div>
                </div>
                </Grid>
                <Grid xs={2} style={{display:'flex', marginTop:'20px'}}>
                    <div style={{fontWeight:'bold'}}>{(item.price)*(item.count)}₴</div>
                </Grid>
                <Grid xs={1} style={{display:'flex', alignItems:'center', justifyContent:'center'}}>
                    <FaTrash className='delete-icon' onClick={()=>removeItem(item.id,true)}/>   
                </Grid>
            </Grid>
        </div>
    )
};
