import React from 'react'
import { FaPhoneAlt } from "react-icons/fa";
import { SlEnvolopeLetter } from "react-icons/sl";
import PhoneLink from './PhoneLink';
import { FaSearchLocation } from "react-icons/fa";

export const Contacts = () => {

    return (
        <div > 
            <div style={{fontSize:'23px', display:'flex', justifyContent:'center', fontWeight:'600', width:'100%'}}>Контакти</div>
            <div className='contacts'>
                <div className='icon-phone' style={{marginBottom:'10px'}}> 
                <FaPhoneAlt />
                <PhoneLink phoneNumber='+380974566344' /> 
                </div>
                <div className='icon-letter' style={{marginBottom:'30px'}}> 
                <SlEnvolopeLetter /> 
                <a className='email-style' href='mailto:daruna.d04@gmail.com'> flowersland@example.com </a>
                </div>
            </div>
            <div className='how-to-find-us' style={{fontSize:'16px'}}>
                <div>
                <FaSearchLocation style={{fontSize:'18px'}}/> Ми знаходимось на вул. П. Куліша 18
                </div>
                <img onClick={() => window.open("https://www.google.com/maps/place/%D0%B2%D1%83%D0%BB%D0%B8%D1%86%D1%8F+%D0%9F%D0%B0%D0%BD%D1%82%D0%B5%D0%BB%D0%B5%D0%B9%D0%BC%D0%BE%D0%BD%D0%B0+%D0%9A%D1%83%D0%BB%D1%96%D1%88%D0%B0,+18,+%D0%9B%D1%8C%D0%B2%D1%96%D0%B2,+%D0%9B%D1%8C%D0%B2%D1%96%D0%B2%D1%81%D1%8C%D0%BA%D0%B0+%D0%BE%D0%B1%D0%BB%D0%B0%D1%81%D1%82%D1%8C,+79000/@49.8473699,24.0238607,19z/data=!4m6!3m5!1s0x473add0ca7cc4e89:0x97fdc94ed069456e!8m2!3d49.8472722!4d24.0245401!16s%2Fg%2F1tcv9nnz", "_blank")} src='/img/map.jpg' alt='map'/>                  
                <div style={{fontSize:'16px', marginTop:'10px', textAlign:'center', color:'#797979'}}>
                    <br></br>
                    Графік роботи: щодня 8:00 - 21:00
                </div>     
            </div>
            
        </div>
    );
};
