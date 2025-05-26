function genetic_algorithm_lanchester
    % Кількість коефіцієнтів у базисній функції
    m = 4;
    
    % Границя для коефіцієнтів
    lb = -10 * ones(1, 2 * m);
    ub = 10 * ones(1, 2 * m);
    
    % Опції генетичного алгоритму
    mutation_rate = 0.08;
    crossover_fraction = 0.85;
    
    options = optimoptions('ga', ...
                           'Display', 'iter', ...
                           'PlotFcn', @gaplotbestf, ...
                           'MutationFcn', {@mutationgaussian, mutation_rate}, ...
                           'CrossoverFraction', crossover_fraction);

    tic;

    % Оптимізація з використанням вбудованого генетичного алгоритму
    [optimal_coeffs, ~] = ga(@(coeffs) lanchester_fitness(coeffs, m), 2 * m, [], [], [], [], lb, ub, [], options);
    
    % Кінець вимірювання часу
    elapsed_time = toc;
    fprintf('Час виконання алгоритму: %.2f секунд\n', elapsed_time);

    % Побудова графіка
    tspan = linspace(0, 10, 1000);  % Збільшено кількість точок для більш точного графіка
    L1 = optimal_coeffs(1:m);
    L2 = optimal_coeffs(m+1:2*m);
    Phi = @(t) [ones(size(t)); t; t.^2; t.^3];
    y1 = @(t) L1 * Phi(t);
    y2 = @(t) L2 * Phi(t);

    % Знаходження точки, де одна зі сторін досягає нуля
    t_zero = 0;
    for t = tspan
        if y1(t) <= 0 || y2(t) <= 0
            t_zero = t;
            break;
        end
    end
    tspan = linspace(0, t_zero, 100);  % Перевизначення tspan до точки t_zero

    figure;
    plot(tspan, y1(tspan), 'b', 'DisplayName', 'y1(t)');
    hold on;
    plot(tspan, y2(tspan), 'r', 'DisplayName', 'y2(t)');
    xlabel('Time t');
    ylabel('Number of Troops');
    legend;
    title('Change in Number of Troops Over Time');
    hold off;
end

function score = lanchester_fitness(coeffs, m)
    % Розбиття коефіцієнтів на два набори для y1 і y2
    L1 = coeffs(1:m);
    L2 = coeffs(m+1:2*m);
    
    % Часовий інтервал для оцінки
    tspan = linspace(0, 10, 10);  % Збільшено кількість точок для більш точного оцінювання
    Phi = @(t) [ones(size(t)); t; t.^2; t.^3];
    
    % Аналітичні розв'язки
    y1 = @(t) L1 * Phi(t);
    y2 = @(t) L2 * Phi(t);
    
    % Похідні
    dy1_dt = @(t) -0.3 * y2(t);
    dy2_dt = @(t) -0.5 * y1(t);
    
    % Оцінка придатності на основі середньоквадратичної помилки
    error_y1 = mean((gradient(y1(tspan), tspan) - dy1_dt(tspan)).^2);
    error_y2 = mean((gradient(y2(tspan), tspan) - dy2_dt(tspan)).^2);
    
    % Початкові умови
    initial_condition_error = (y1(0) - 20)^2 + (y2(0) - 60)^2;
    
    % Загальний результат
    score = error_y1 + error_y2 + initial_condition_error; % Мінімізуємо помилку
end
