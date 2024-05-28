import { Checkbox, FormControl, FormControlLabel, FormGroup, Grid, InputLabel, MenuItem, Select, Typography } from '@mui/material';
import React, {useState} from 'react'
import { HiOutlineLocationMarker } from "react-icons/hi";
import AlertDialog from './Final';



export function LastStepToOrder({setPage,items,removeItem}) {

  const [formData, setFormData] = useState({
    name: '',
    phone: '+380',
    email: '',
    street: '',
    house: '',
    flat: '',
    comment: '',
    needCall: false,
    selectedDate: '',
    payWhenRecieve: false,
    payNow: false,
  });



  const handleSubmit = (e) => {
    e.preventDefault();
    if (!formData.email.includes("@") || !formData.phone.match(/^\+38\d{9}$/)) {
      alert("Неправильні дані. Будь ласка, перевірте їх та спробуйте знову.");
      return;
    }
  };

  const [isDelivery,setIsDelivery] = useState(false);

  const handleChange = (e) => {
    setFormData((prevFormData) => ({
      ...prevFormData,
      selectedDate: e.target.value,
    }));
  }

  const [isChecked, setIsChecked] = useState(false);

  const handleCheckboxChange = (event) => {
    setIsChecked(event.target.checked);
  };


  return (
    <div className='sign-up'>
      <div style={{fontSize:'23px', fontWeight:'600', marginBottom:'40px', justifyContent:'center', display:'flex'}}>
        Оформлення замовлення
      </div>
      <Grid container className="forms-container">
        <Grid xs={4}>
        <form className="form" onSubmit={handleSubmit}>
           <div style={{fontWeight:'600', fontSize:'20px'}}> 
            <div className='circle' >1</div>
            <div >Ваші дані</div>
            </div> <br/>
          <label>     
            <input
              placeholder="Ім'я"
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            />

            <input
              placeholder='Номер телефону'
              type='tel'
              value={formData.phone}
              onChange={(e) => setFormData({ ...formData, phone: e.target.value })}
              onBlur={(e) => {
                console.log(111)
                
              }}
              required    
            />
            <input
              placeholder='Електронна пошта'
              type="email"
              value={formData.email}
              onChange={(e) => setFormData({ ...formData, email: e.target.value })}
            />
            
          </label>                
          <br /> <br />
          <textarea className='comment' placeholder='Коментар до замовлення' value={formData.comment}
              onChange={(e) => setFormData({ ...formData, comment: e.target.value })}>
            </textarea><br /> <br />
        </form>
        </Grid>
        <Grid xs={4} style={{display:'flex', justifyContent:'center'}}>
        <div className="form">
        <div style={{fontWeight:'600', fontSize:'20px'}}> 
            <div className='circle'>2</div>
            <div>Доставка</div>
            </div> <br/>
            <div className='is-delivery'
            onClick={()=>setIsDelivery(!isDelivery)}> {isDelivery ? "Вибрати доставку кур'єром": 'Вибрати самовивіз із магазину'}
            </div>
            {isDelivery ? 
            <div> <HiOutlineLocationMarker style={{fontSize:'22px'}}/>FlowersLand (м. Львів, вул. П.Куліша 18) <br /> <br />
            <FormControl style={{width:'80%', borderRadius:'50px', textAlign:'center'}}>
            <InputLabel >Дата</InputLabel>
            <Select style={{borderRadius:'50px',height:'45px'}}
                value={formData.selectedDate}
                label="Date"
                // onChange={(e) => setFormData({ ...formData, flat: e.target.value })}
                onChange={handleChange}
            >
                <MenuItem value={'На сьогодні (12.06.2023)'}>На сьогодні (12.06.2023)</MenuItem>
                <MenuItem value={'На завтра (13.06.2023)'}>На завтра (13.06.2023)</MenuItem>
                <MenuItem value={'14.06.2023'}>14.06.2023</MenuItem>
                <MenuItem value={'15.06.2023'}>15.06.2023</MenuItem>
            </Select>
            </FormControl> <br />
            <div style={{fontSize:'10px'}}> 
                *Для замовлення в іншу дату прошу зателефонувати за номером, що вказаний в розділі "Контакти"
            </div>
            </div>: 
            <div>
                <form>                   
                    <input style={{width:'80%'}}
                    placeholder="Вулиця"
                    type="text"
                    value={formData.street}
                    onChange={(e) => setFormData({ ...formData, street: e.target.value })}
                    />   
                    <div>
                    <input style={{width:'33%', marginRight:'15px'}}
                    placeholder="Дім"
                    type="text"   
                    maxLength={5}
                    value={formData.house}
                    onChange={(e) => setFormData({ ...formData, house: e.target.value })}
                    />   
                    <input style={{width:'30%'}}
                    placeholder="Квартира"
                    type="number"                   
                    maxLength={5}
                    value={formData.flat}
                    onChange={(e) => setFormData({ ...formData, flat: e.target.value })}

                    />   
                    </div>
                </form> <br />
                <FormControl style={{width:'93%', borderRadius:'50px'}}>
                <InputLabel >Дата</InputLabel>
                <Select style={{borderRadius:'50px', height:'45px', textAlign:'center', justifyContent:'center'}}
                    value={formData.selectedDate}
                    label="Date"
                    onChange={handleChange}
                >
                <MenuItem value={'На сьогодні (12.06.2023)'}>На сьогодні (12.06.2023)</MenuItem>
                <MenuItem value={'На завтра (13.06.2023)'}>На завтра (13.06.2023)</MenuItem>
                <MenuItem value={'14.06.2023'}>14.06.2023</MenuItem>
                <MenuItem value={'15.06.2023'}>15.06.2023</MenuItem>
                </Select>
                </FormControl> <br />
                <div style={{fontSize:'10px'}}> 
                *Для замовлення в іншу дату прошу зателефонувати за номером, що вказаний в розділі "Контакти"
            </div>
            </div>
                 }
        </div>
        </Grid>
        
        <Grid xs={4}>
        <div className="form">
        <div style={{fontWeight:'600', fontSize:'20px'}}> 
            <div className='circle'>3</div>
            <div >Оплата</div>
            </div> <br/>
        <div>
        <FormGroup >
            <FormControlLabel 
                 control={<Checkbox checked={formData.payWhenRecieve} onChange={(e) => setFormData({ ...formData, payWhenRecieve: e.target.checked })} />} 
                 label={<span style={{ fontFamily: 'Montserrat, sans-serif' }}>{ "Оплата при отриманні" }</span>} 
            />
        </FormGroup> <br />
        <FormGroup>
      <FormControlLabel
        control={<Checkbox 
            checked={formData.payNow}
            onChange={(e) => {
              const isChecked = e.target.checked;
              setFormData((prevFormData) => ({
                ...prevFormData,
                payNow: isChecked,
              }));
              handleCheckboxChange(e); 
            }}
          />}
        label={
          <Typography style={{ fontFamily: "Montserrat, sans-serif" }}>
            {isChecked ? (
              <a href="https://next.privat24.ua/money-transfer/card" style={{textDecoration:'underline'}}>Натисніть для переходу на сторінку реквізитів</a>
            ) : (
              "Оплата за реквізитами онлайн"
            )}
          </Typography>
        }
      />
    </FormGroup>
        </div>
        </div>
        </Grid>
      </Grid>
    <FormGroup>
      <FormControlLabel control={<Checkbox  checked={formData.needCall} onChange={(e) => setFormData({ ...formData, needCall: e.target.checked })}  />} 
      label={<span style={{ fontFamily: 'Montserrat, sans-serif' }}>{ "Мені не потрібно телефонувати для підтвердження" }</span>}   
      />
    </FormGroup>

    <AlertDialog 
     setPage={setPage} formData={formData} items={items} removeItem={removeItem}/> 
    </div>
  )
}


