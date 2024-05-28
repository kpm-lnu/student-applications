import * as React from 'react';
import Dialog from '@mui/material/Dialog';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import DialogTitle from '@mui/material/DialogTitle';
import TaskAltOutlinedIcon from '@mui/icons-material/TaskAltOutlined';


export default function AlertDialog ({setPage,formData, items,removeItem}) {

  const [open, setOpen] = React.useState(false);

  const sendMessage = (data)=>{
    const selectedItems = items.filter((item)=> item.count);
    console.log(2,selectedItems)
    fetch('http://localhost:3000/send_order',{
      method: 'POST', 
      headers: {
        'Content-Type': 'application/json' 
      },
      body: JSON.stringify({...data,selectedItems}) 
    })
  .then(response => response.json())
  .catch(error => {
    console.error('Помилка:', error);
  });
  }

  const handleClickOpen = () => {
    setOpen(true);
    sendMessage(formData);
    removeItem(false,false,true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  return (
    <div>
        <div style={{alignItems:'center', display:'flex', justifyContent:'center', width: '100%'}}>
            <button type="submit" className='button-submit' onClick={handleClickOpen}>Оформлення замволення</button>
        </div>
      <Dialog
        open={open}
        onClose={handleClose}
      >
        <DialogTitle >
        {<h3 style={{ fontFamily: 'Montserrat, sans-serif' }}>{ "Ви успішно здійснили замволення!" }</h3>} 
        </DialogTitle>
        <DialogContent style={{width: '200px', height: '200px'}}>
            <div style={{display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%'}}>
              <TaskAltOutlinedIcon style={{fontSize:'150px', color:'#5fa36a'}}/>
            </div>
        </DialogContent>
        <DialogActions>
          <div>
            <button type="submit" className='button-final'  onClick={()=>{setPage(6)}}> Готово </button>
          </div>
        </DialogActions>
      </Dialog>
    </div>
  );
}