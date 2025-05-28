function diskret_system()

% Задаємо початкові дані
A = [0 0.7 0.01;
     0.3 0 0.08];

B = [0.1 0.05;
     0.02 0.07;
     0.15 0.15];

% Кількість солдат на початку
initial_A = [38; 48];  % Кількість солдат у армії A
initial_B = [35; 12; 50];  % Кількість солдат у армії B

% Час моделювання
time = 0:0.1:10;

% Використаємо функцію ode45 для розв'язання системи диференціальних рівнянь
[t, y] = ode45(@(t, y) lanchesterEquations(t, y, A, B), time, [initial_A; initial_B]);

% Побудуємо графіки
figure;
plot(t, y(:, 1), '-b', 'LineWidth', 2, 'DisplayName', 'літаки');
hold on;
plot(t, y(:, 2), '--b', 'LineWidth', 2, 'DisplayName', 'танки');
plot(t, y(:, 3), 'r', 'LineWidth', 2, 'DisplayName', 'артилерія');
plot(t, y(:, 4), '--r', 'LineWidth', 2, 'DisplayName', 'ППО');
plot(t, y(:, 5), '-.r', 'LineWidth', 2, 'DisplayName', 'піхота');
xlabel('Час');
ylabel('Кількість військ');
legend('show');
title('Моделювання бойових дій за допомогою дискретних рівнянь');
grid on;

end

% Функція, яка визначає систему диференціальних рівнянь Ланчестера
function dydt = lanchesterEquations(~, y, A, B)
    dydt = zeros(5, 1);

    % Втрати армії A
    dydt(1) = -sum(A(1, :) .* max(y(3:5), 0)');
    dydt(2) = -sum(A(2, :) .* max(y(3:5), 0)');

    % Втрати армії B
    dydt(3) = -sum(B(1, :) .* max(y(1:2), 0)');
    dydt(4) = -sum(B(2, :) .* max(y(1:2), 0)');
    dydt(5) = -sum(B(3, :) .* max(y(1:2), 0)');

    % Упевнитися, що кількість солдат не опускається нижче нуля
    dydt = max(dydt, -y);
end

