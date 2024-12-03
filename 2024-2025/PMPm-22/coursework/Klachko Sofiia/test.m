clear all;
clc;
% Параметри моделі
r = 0.1;       % Швидкість росту
K = 1000;      % Максимальна кількість клітин
N0 = 10;       % Початкова кількість клітин
t_max = 50;    % Максимальний час
h = 1;         % Крок інтегрування

% Функція для похідної
dN_dt = @(N) r * N * (1 - N / K);

% Точний аналітичний розв'язок
N_exact = @(t) (K * N0 * exp(r * t)) ./ (K + N0 * (exp(r * t) - 1));

% Часовий вектор
t = 0:h:t_max;

% Точні значення
N_exact_values = N_exact(t);

% Метод Ейлера
N_euler = zeros(size(t));
N_euler(1) = N0;
for i = 1:length(t)-1
    N_euler(i+1) = N_euler(i) + h * dN_dt(N_euler(i));
end

% Метод Рунге-Кутта 4-го порядку
N_rk4 = zeros(size(t));
N_rk4(1) = N0;
for i = 1:length(t)-1
    k1 = dN_dt(N_rk4(i));
    k2 = dN_dt(N_rk4(i) + h/2 * k1);
    k3 = dN_dt(N_rk4(i) + h/2 * k2);
    k4 = dN_dt(N_rk4(i) + h * k3);
    N_rk4(i+1) = N_rk4(i) + h/6 * (k1 + 2*k2 + 2*k3 + k4);
end

% Обчислення похибки
error_euler = abs(N_exact_values - N_euler);
error_rk4 = abs(N_exact_values - N_rk4);

% Виведення результатів у командне вікно
disp('Часовий момент | Точний розв’язок | Метод Ейлера | Похибка (Ейлер) | Метод Рунге-Кутта | Похибка (РК)');
disp('----------------------------------------------------------------------------------------------');
for i = 1:length(t)
    fprintf('%14.2f | %15.5f | %12.5f | %16.5f | %19.5f | %15.5f\n', ...
        t(i), N_exact_values(i), N_euler(i), error_euler(i), N_rk4(i), error_rk4(i));
end

% Графік 1: Динаміка росту
figure;
plot(t, N_exact_values, 'k-.', 'LineWidth', 1.5, 'DisplayName', 'Точний розв’язок');
hold on;
plot(t, N_euler, 'b-', 'LineWidth', 1, 'DisplayName', 'Метод Ейлера');
plot(t, N_rk4, 'r-', 'LineWidth', 1, 'DisplayName', 'Метод Рунге-Кутта');
xlabel('Час, t');
ylabel('Кількість клітин, N(t)');
legend('Location', 'best');
title('Порівняння чисельних методів з точним розв’язком');
grid on;

% Графік 2: Похибки
figure;
plot(t, error_euler, 'b--o', 'LineWidth', 1, 'DisplayName', 'Похибка методу Ейлера');
hold on;
plot(t, error_rk4, 'r-.x', 'LineWidth', 1, 'DisplayName', 'Похибка методу Рунге-Кутта');
xlabel('Час, t');
ylabel('Похибка');
legend('Location', 'best');
title('Порівняння похибок чисельних методів');
grid on;

