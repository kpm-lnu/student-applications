% Початкові сили
Greeks = 6300; % Сили Греків
Persians = 160000; % Сили Персів

% Ефективність
a = 2.5; % Ефективність Греків
b = 0.7; % Ефективність Персів

% Максимальна кількість військових, що можуть одночасно брати участь у битві
max_battle = 100;

% Кроки часу
T = 96;

% Ініціалізація масивів для зберігання сил протягом часу
Greek_strength = zeros(1, T);
Persian_strength = zeros(1, T);

% Початкові сили
Greek_strength(1) = Greeks;
Persian_strength(1) = Persians;

% Симуляція битви
for t = 2:T
    % Визначаємо кількість військових, що беруть участь у битві
    Greek_battle = min(Greek_strength(t-1), max_battle);
    Persian_battle = min(Persian_strength(t-1), max_battle);
    
    % Оновлюємо сили на основі кількості військових, що брали участь у битві
    Greek_strength(t) = Greek_strength(t-1) - b * Persian_battle;
    Persian_strength(t) = Persian_strength(t-1) - a * Greek_battle;
    
    % Якщо сила однієї із сторін опускається нижче нуля, встановлюємо її в нуль
    if Greek_strength(t) < 0
        Greek_strength(t) = 0;
    end
    if Persian_strength(t) < 0
        Persian_strength(t) = 0;
    end
end

% Виведення початкового та останнього значення сил
fprintf('Початкове значення Греків: %d\n', Greek_strength(1));
fprintf('Останнє значення Греків: %d\n', Greek_strength(T));
fprintf('Початкове значення Персів: %d\n', Persian_strength(1));
fprintf('Останнє значення Персів: %d\n', Persian_strength(T));

% Побудова результатів
figure;
plot(1:T, Greek_strength, 'b', 1:T, Persian_strength, 'r','LineWidth',2);
xlabel('Час');
ylabel('Чисельність війська');
legend('Грецька армія', 'Перська армія');
title('Симуляція битви при Фермопілах');
