// Отримання елементів DOM
const cartButton = document.getElementById('cartButton');
const cartModal = document.getElementById('cartModal');
const closeButton = document.getElementById('closeButton');
const cartItemsElement = document.getElementById('cartItems');
const totalPriceElement = document.getElementById('totalPrice');
const clearCartButton = document.getElementById('clearCartButton');

// Обробник події для відкриття модального вікна
cartButton.addEventListener('click', function () {
    cartModal.style.display = 'block'; // Показуємо модальне вікно
});

// Обробник події для закриття модального вікна
closeButton.addEventListener('click', function () {
    cartModal.style.display = 'none'; // Ховаємо модальне вікно
});

// Отримання даних з локального сховища
let cartItems = JSON.parse(localStorage.getItem('cartItems')) || [];

// Відображення товарів у корзині
function displayCartItems() {
    cartItemsElement.innerHTML = ''; // Очистка попереднього вмісту
    let totalPrice = 0;

    cartItems.forEach(function (item) {
        const productElement = document.createElement('div');
        productElement.classList.add('product');
        productElement.innerHTML = `
      <img src="${item.image}" alt="${item.name}" />
      <h3>${item.name}</h3>
      <p>Ціна: $${item.price}</p>
      <p>Кількість: ${item.quantity}</p>
    `;
        cartItemsElement.appendChild(productElement);

        const itemPrice = item.price * item.quantity;
        totalPrice += itemPrice;
    });

    totalPriceElement.textContent = `Загальна вартість: $${totalPrice.toFixed(2)}`;
}

// Оновлення даних у локальному сховищі
function updateCartItems() {
    localStorage.setItem('cartItems', JSON.stringify(cartItems));
}

// Очищення корзини
function clearCart() {
    cartItems = [];
    updateCartItems();
    displayCartItems();
}

// Додавання товару до корзини
function addToCart(name, price, image) {
    let item = cartItems.find(item => item.name === name);
    if (item) {
        item.quantity += 1;
    } else {
        item = { name, price, image, quantity: 1 };
        cartItems.push(item);
    }
    updateCartItems();
    displayCartItems();
}

// Виклик функції для відображення товарів у корзині
displayCartItems();

// Обробник події для кнопки "Очистити корзину"
clearCartButton.addEventListener('click', clearCart);

function goToHomePage() {
    window.location.href = "../index.html";
}

function goToContactsPage() {
    window.location.href = "../contacts/contacts.html";
  }

document.getElementById('checkoutButton').addEventListener('click', function() {
    // Отримуй дані для замовлення
    var name = prompt('Введіть ваше ім\'я:');
    var email = prompt('Введіть вашу електронну пошту:');
    var address = prompt('Введіть вашу адресу доставки:');
  // Перевіряю, чи користувач ввів адресу
  if (name && email && address) {
    alert('Дякуємо за замовлення!');
    clearCart();
    window.location.href = "../index.html";
  }
  else{
    alert('Будь ласка, заповніть всі поля для оформлення замовлення.');
  }
  });






