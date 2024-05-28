const TelegramApi = require("node-telegram-bot-api");

const TOKEN = '6030910125:AAGLExmf-T9I5IUpXEFJLqh8H9O5vMLjg1Q';
const bot = new TelegramApi(TOKEN, { polling: true });



const express = require('express');
const cors = require('cors');
const app = express();
const port = 3000;
app.use(cors({
	origin: 'http://localhost:3001', 
	methods: 'GET, POST', 
  }));
app.use(express.json());



app.post('/send_order', (req, res) => {
	console.log(req);
	let a = '🟩\n\n\n\n'
 Object.keys(req.body).forEach((key) => {
	if(key === 'selectedItems'){
		req.body[key].forEach((flower)=> a= a+ `\n🌷${flower.title} x${flower.count}\nСума:${flower.price*flower.count}`)
	}
	else {
		a = a+ `${key}: ${req.body[key]} \n`;
	}
} );
  res.send({ message: 'Привіт з бекенду!' });
  bot.sendMessage('404348064', a+'\n\n\n\n🟩');
});

app.listen(port, () => {
  console.log(`Сервер запущено на порту ${port}`);
});