function covid19_model_with_sliders
    % Завантаження CSV-файлу
    [file, path] = uigetfile('*.csv', 'Select the CSV file');
    if isequal(file, 0)
        disp('User selected Cancel');
        return;
    else
        filename = fullfile(path, file);
        disp(['User selected ', filename]);
    end

    % Завантаження реальних даних
    real_data = readtable(filename, 'VariableNamingRule', 'preserve');
    disp(real_data);

    % Початкові параметри моделі
    beta = 0.3;
    sigma = 0.1;
    gamma = 0.05;
    mu = 0.01;
    mu_I = 0.02;
    delta = 0.01;
    v = 0.02;

    % Створення GUI
    fig = figure('Name', 'SEIR Model with Sliders', 'NumberTitle', 'off', 'Position', [100 100 1200 600]);

    % Слайдери для параметрів
    uicontrol('Style', 'text', 'Position', [50 550 150 20], 'String', 'Ймовірність зараження (beta)');
    sldBeta = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', beta, 'Position', [50 520 150 20]);
    addlistener(sldBeta, 'Value', 'PostSet', @(src, event) updatePlot());

    uicontrol('Style', 'text', 'Position', [50 480 150 20], 'String', 'Швидкість переходу в інфіковані (sigma)');
    sldSigma = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', sigma, 'Position', [50 450 150 20]);
    addlistener(sldSigma, 'Value', 'PostSet', @(src, event) updatePlot());

    uicontrol('Style', 'text', 'Position', [50 410 150 20], 'String', 'Швидкість одужання (gamma)');
    sldGamma = uicontrol('Style', 'slider', 'Min', 0, 'Max', 1, 'Value', gamma, 'Position', [50 380 150 20]);
    addlistener(sldGamma, 'Value', 'PostSet', @(src, event) updatePlot());

    uicontrol('Style', 'text', 'Position', [50 340 150 20], 'String', 'Народжуваність/смертність (mu)');
    sldMu = uicontrol('Style', 'slider', 'Min', 0, 'Max', 0.1, 'Value', mu, 'Position', [50 310 150 20]);
    addlistener(sldMu, 'Value', 'PostSet', @(src, event) updatePlot());

    uicontrol('Style', 'text', 'Position', [50 270 150 20], 'String', 'Смертність від інфекції (mu_I)');
    sldMu_I = uicontrol('Style', 'slider', 'Min', 0, 'Max', 0.1, 'Value', mu_I, 'Position', [50 240 150 20]);
    addlistener(sldMu_I, 'Value', 'PostSet', @(src, event) updatePlot());

    uicontrol('Style', 'text', 'Position', [50 200 150 20], 'String', 'Додаткове одужання (delta)');
    sldDelta = uicontrol('Style', 'slider', 'Min', 0, 'Max', 0.1, 'Value', delta, 'Position', [50 170 150 20]);
    addlistener(sldDelta, 'Value', 'PostSet', @(src, event) updatePlot());

    uicontrol('Style', 'text', 'Position', [50 130 150 20], 'String', 'Швидкість вакцинації (v)');
    sldV = uicontrol('Style', 'slider', 'Min', 0, 'Max', 0.1, 'Value', v, 'Position', [50 100 150 20]);
    addlistener(sldV, 'Value', 'PostSet', @(src, event) updatePlot());

    % Початкові умови
    S0 = 1000;
    E0 = 1;
    I0 = 0;
    R0 = 0;

    % Початковий вектор стану
    initial_conditions = [S0, E0, I0, R0];

    % Час моделювання
    tspan = [0 50];

    % Функція SEIR моделі
    function dydt = seir_model(t, y, beta, sigma, gamma, mu, mu_I, delta, v)
        S = y(1);
        E = y(2);
        I = y(3);
        R = y(4);

        dSdt = mu * (S + E + I + R) - beta * S * I - mu * S - v * S;
        dEdt = beta * S * I - sigma * E - mu * E;
        dIdt = sigma * E - gamma * I - mu * I - mu_I * I;
        dRdt = gamma * I - mu * R + delta * E + v * S;

        dydt = [dSdt; dEdt; dIdt; dRdt];
    end

    % Функція для оновлення графіку
    function updatePlot()
        beta = get(sldBeta, 'Value');
        sigma = get(sldSigma, 'Value');
        gamma = get(sldGamma, 'Value');
        mu = get(sldMu, 'Value');
        mu_I = get(sldMu_I, 'Value');
        delta = get(sldDelta, 'Value');
        v = get(sldV, 'Value');

        [t, y] = ode45(@(t, y) seir_model(t, y, beta, sigma, gamma, mu, mu_I, delta, v), tspan, initial_conditions);

        % Очистка та оновлення графіку
        subplot('Position', [0.3 0.1 0.65 0.8]);
        cla;
        hold on;
        plot(t, y(:,1), 'b', 'DisplayName', 'Susceptible (Прогноз)');
        plot(t, y(:,2), 'm', 'DisplayName', 'Exposed (Прогноз)');
        plot(t, y(:,3), 'r', 'DisplayName', 'Infected (Прогноз)');
        plot(t, y(:,4), 'g', 'DisplayName', 'Recovered (Прогноз)');

        % Додавання реальних даних до графіку
        plot(real_data.Time, real_data.Susceptible, 'b--', 'DisplayName', 'Susceptible (Реальні)');
        plot(real_data.Time, real_data.Exposed, 'm--', 'DisplayName', 'Exposed (Реальні)');
        plot(real_data.Time, real_data.Infected, 'r--', 'DisplayName', 'Infected (Реальні)');
        plot(real_data.Time, real_data.Recovered, 'g--', 'DisplayName', 'Recovered (Реальні)');

        xlabel('Час');
        ylabel('Населення');
        legend;
        title('SEIR Модель з порівнянням реальних даних');
        hold off;
    end

    % Початковий виклик оновлення графіку
    updatePlot();
end

