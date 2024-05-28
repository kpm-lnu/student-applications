import React from "react";
import {Footer} from "./components/Footer";
import { About } from "./components/About";
import { Contacts } from "./components/Contacts";
import {Header} from "./components/Header";
import { Main } from "./components/Main";
import { MakeOrder } from "./components/MakeOrder";
import { LastStepToOrder } from "./components/LastStepToOrder";
import { useEffect, useState } from "react";
import { db } from "./config/firebase";
import {
  getDocs,
  collection,
} from 'firebase/firestore';
import Review from "./components/Review";




export const App = () => {

  const [allItems,setAllItems] = useState([]);

  const [defaultItems,setDefaultItems] = useState([]);

  const defaultItemsCollectionRef = collection(db, "defaultItems");

  useEffect(() => {
    const getFlowerList = async () => {
        try {
          const data = await getDocs(defaultItemsCollectionRef);
          const filteredData = data.docs.map((doc) => ({
            ...doc.data(),
            id: doc.id,
          }));
          setAllItems(filteredData);
          setDefaultItems(filteredData);
        } 
        catch (err) {
          console.error(err);
        }
      };

    getFlowerList();
  }, []);

  // const defaultItems = [
  //   {
  //     id: 1,
  //     title: 'Букет з троянд',
  //     img: '51rose.jpg',
  //     desc: 'Букет з 51 троянди',
  //     category: 'rose',
  //     price: '1700',
  //     count:0,
  //   },
  //   {
  //     id: 2,
  //     title: 'Букет з ромашок',
  //     img: '29daisies.jpg',
  //     desc: 'Букет з 29 ромашок',
  //     category: 'daisies',
  //     price: '1100',
  //     count:0,
  //   },
  //   {
  //     id: 3,
  //     title: 'Букет з полявих ромашок',
  //     img: 'daisies.jpg',
  //     desc: 'Букет з 35 польових ромашок',
  //     category: 'daisies',
  //     price: '899',
  //     count:0,
  //   },
  //   {
  //     id: 4,
  //     title: 'Букет з оранджевих тюльпанів',
  //     img: 'orangetulips.jpg',
  //     desc: 'Букет з 51 троянди',
  //     category: 'tulips',
  //     price: '789',
  //     count:0,
  //   },
  //   {
  //     id: 5,
  //     title: 'Букет з рожевих тюльпанів',
  //     img: 'pinktulips.jpg',
  //     desc: 'Букет з 51 троянди',
  //     category: 'tulips',
  //     price: '2500',
  //      count:0,
  //   },
  //   {
  //     id: 6,
  //     title: 'Букет з фіолетових тюльпанів',
  //     img: 'purpletulips.jpg',
  //     desc: 'Букет з 51 троянди',
  //     category: 'tulips',
  //     price: '1349',
  //     count:0,
  //   },
  //   {
  //       id: 7,
  //       title: 'Букет з ромашок',
  //       img: '29daisies.jpg',
  //       desc: 'Букет з 29 ромашок',
  //       category: 'daisies',
  //       price: '1100',
  //       count:0,
  //     },
  // ]

  const [page, setPage] = useState(0);

  

  const addItem = (id) =>{
    const newItems =[];
    allItems.map((item)=>  newItems.push(item.id === id ? ({...item,count: item.count + 1 }) : item ) );
    setAllItems(newItems);
  }

  const removeItem = (id,total, all) =>{
    if(all) {
    setAllItems(defaultItems);

    }
    else{
    const newItems =[];
    allItems.map((item)=>  newItems.push(item.id === id ? ({...item,count: total ? 0 : item.count - 1 }) : item ) );
    setAllItems(newItems);
    }
  }

 

const pageToShow = ()=> {
  switch(page){
  case 1: return <About />;
  case 2: return <Contacts />; 
  case 3: return <Review />
  // case 3: return <LastStepToOrder setPage={setPage} items={allItems} removeItem={removeItem}/>;
  case 4: return <MakeOrder items={allItems} removeItem={removeItem} addItem={addItem} setPage={setPage} />
  case 5: return <LastStepToOrder setPage={setPage} items={allItems} removeItem={removeItem}/>;
  // case 6: return <Main addItem={addItem} allItems={allItems}/>
   default: return <Main addItem={addItem} allItems={allItems}/>
}};


    return (
      <div style={{position:'relative'}}>
        <Header removeItem={removeItem} addItem={addItem} setPage={setPage} items={allItems} />
       <div  className="wrapper">{ pageToShow()}</div> 
       
       <Footer />
      </div>
    );
  }

