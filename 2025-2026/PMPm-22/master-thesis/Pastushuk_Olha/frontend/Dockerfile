FROM node:20-slim

WORKDIR /app

# Встановлюємо залежності
COPY package*.json ./
RUN npm install

# Копіюємо решту коду
COPY . .

EXPOSE 3000

CMD ["npm", "start"]