% Оголошення глобальних змінних
global N mu alpha beta delta v

% Задаємо параметри моделі SEIR
N = 269900;        % Загальна чисельність населення
S0 = 162919;        % Початкова чисельність сприйнятливих
E0 = 7587;        % Початкова чисельність підозрюваних
I0 = 13638;        % Початкова чисельність інфікованих
R0 = 85756;        % Початкова чисельність одужаних

mu = 0.0174;       % Народжуваність/смертність
alpha = 0.24;  % Коефіцієнт передачі від сприйнятливих до підозрюваних
beta = 0.786;     % Коефіцієнт передачі від підозрюваних до інфікованих
delta = 0.0714;    % Коефіцієнт одужання від інфекції
v = 0.0003;        % Коефіцієнт вакцинації

% Відрізок часу
T = 365;           % Кількість днів для моделювання
dt = 1;            % Крок часу
tspan = 0:dt:T;    % Проміжок часу

% Вектор початкових умов
initial_conditions = [S0, E0, I0, R0];

% Визначення системи диференціальних рівнянь SEIR
function dXdt = SEIR(t, X)
    global N mu alpha beta delta v
    S = X(1);
    E = X(2);
    I = X(3);
    R = X(4);
    
    % Рівняння системи SEIR
    dSdt = mu * (N - S) - alpha * S * I - v * S;
    dEdt = alpha * S * I - beta * E - mu * E;
    dIdt = beta * E - delta * I - mu * I;
    dRdt = delta * I - mu * R + v * S;
    
    % Повернення результатів у вигляді вектора
    dXdt = [dSdt; dEdt; dIdt; dRdt];
end

% Розв'язання системи за допомогою ode45
[t, X] = ode45(@SEIR, tspan, initial_conditions);

% Витягування рішень для кожної популяції
S = X(:, 1);
E = X(:, 2);
I = X(:, 3);
R = X(:, 4);

% Побудова графіків результатів
figure;
plot(t, S, 'b', 'LineWidth', 2); hold on;
plot(t, E, 'm', 'LineWidth', 2);
plot(t, I, 'r', 'LineWidth', 2);
plot(t, R, 'g', 'LineWidth', 2);
xlabel('Час (дні)');
ylabel('Чисельність населення');
legend({'Сприйнятливі', 'Підозрювані', 'Інфіковані', 'Одужані'});
title(['Модель SEIR для COVID-19' ...
       '\newline Параметри: N = ' num2str(N) ...
       ', \mu = ' num2str(mu) ...
       ', \alpha = ' num2str(alpha) ...
       ', \beta = ' num2str(beta) ...
       ', \delta = ' num2str(delta) ...
       ', \nu = ' num2str(v)]);
grid on;
hold off;
