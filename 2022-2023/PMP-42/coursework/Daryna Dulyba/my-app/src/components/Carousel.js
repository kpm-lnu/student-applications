import React from 'react';
import Carousel from 'react-material-ui-carousel'
import { Paper } from '@mui/material'

 function Example(props)
{
    var itemsAbout= [
        {

            image: 'flowers1.webp'
        },
        {

            image: 'flowers2.avif'
        },
        {
            image: 'flowersdelivery.jpeg'
        },
        {
            image: 'flowersworkers1.avif'
        },
        {
            image: 'flowersworkers2.avif'
        }
    ]

    return (
        <Carousel>
            {
                itemsAbout.map( (item, i) => <Item key={i} item={item} /> )
            }
        </Carousel>
    )
}

 function Item(props)
{
    return (
        <Paper className='carousel' elevation={0}>
            <img  src={`./img/${props.item.image}`} alt='fhfj'/>       
        </Paper>
    )
}

export default Example