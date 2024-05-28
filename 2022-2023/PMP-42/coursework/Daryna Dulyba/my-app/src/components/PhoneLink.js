import React from 'react'

function PhoneLink({ phoneNumber }) {
    return (
      <a className='phone-nubmer-style' href={`tel:${phoneNumber}`}>{phoneNumber}</a>
    );
  }

export default PhoneLink