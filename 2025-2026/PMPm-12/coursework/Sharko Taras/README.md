# Sylvester ADI Solver (C# + F# + React)

Система для розв'язання матричного рівняння Сильвестра

AX + XB = C

методом Alternating Direction Implicit (ADI).

## Архітектура

- `backend/src/SylvesterSolver.FSharp`: F# бібліотека з чисельним ADI ядром.
- `backend/src/SylvesterApi`: C# Web API (валідація, DI, API контракти, інтеграція з F#).
- `frontend`: React + TypeScript + Vite UI.

## Принципи якості

- SOLID: розділені контракти, валідація, сервісний шар, інверсія залежностей через DI.
- Чіткі DTO контракти між UI та API.
- Обробка помилок в API (400 для валідації, 500 для неочікуваних винятків).
- Перевірена end-to-end коректність на тестовому кейсі.

## Запуск

### 1) Backend

```bash
dotnet run --project backend/src/SylvesterApi/SylvesterApi.csproj --urls http://localhost:5000
```

OpenAPI (dev):

`http://localhost:5000/openapi/v1.json`

### 2) Frontend

```bash
cd frontend
npm install
npm run dev
```

UI буде доступний на `http://localhost:5173`.

## Налаштування API URL у фронтенді

За замовчуванням використовується `http://localhost:5000`.

За потреби додай файл `.env` у `frontend`:

```bash
VITE_API_BASE_URL=http://localhost:5000
```

## Формат вводу матриць у UI

- Рядки розділяй переносом рядка або `;`
- Елементи в рядку розділяй пробілом або `,`

Приклад:

```text
4, 1
2, 3
```
