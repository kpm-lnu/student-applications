clear all;
close all;

%% Параметри
t0 = 0; tf = 50; h = 0.01;
t = t0:h:tf;
num_steps = length(t);

%% --- Допоміжні функції ---
function y_next = runge_kutta(f, t, y, h, varargin)
    k1 = f(t, y, varargin{:});
    k2 = f(t + h/2, y + h/2 * k1, varargin{:});
    k3 = f(t + h/2, y + h/2 * k2, varargin{:});
    k4 = f(t + h, y + h * k3, varargin{:});
    y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4);
end

function y_next = runge_kutta_scalar(f, t, y, h)
    k1 = f(t, y);
    k2 = f(t + h/2, y + h/2 * k1);
    k3 = f(t + h/2, y + h/2 * k2);
    k4 = f(t + h, y + h * k3);
    y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4);
end

function dy = verhulst_rhs(t, y, r, K, C0, a, b, Cmax)
    N = max(y(1), eps);
    C = max(y(2), eps);
    dN = r * N * (1 - N / K) * (C / (C + C0));
    dC = -a * N * C + b * (Cmax - C);
    dy = [dN; dC];
end

function [N_steady, t_90] = compute_metrics(t, N)
    last_10_percent = floor(0.9 * length(t));
    N_steady = mean(N(last_10_percent:end));
    idx_90 = find(N >= 0.9 * N_steady, 1, 'first');
    if isempty(idx_90)
        t_90 = NaN;
    else
        t_90 = t(idx_90);
    end
end

%% МОДЕЛЬ ВЕРГУЛА
% Базові параметри
r = 0.5; K = 100; C0 = 1; a = 0.01; b = 0.1; Cmax = 1;
N0 = 10; C_init = 1;

y_verhulst = zeros(num_steps, 2);
y_verhulst(1, :) = [N0, C_init];
for i = 1:num_steps-1
    y_verhulst(i+1, :) = runge_kutta(@verhulst_rhs, t(i), y_verhulst(i, :)', h, r, K, C0, a, b, Cmax)';
end
[Ns_v, t90_v] = compute_metrics(t, y_verhulst(:, 1));
printf('Вергула (базовий): N_steady = %.2f, t_90 = %.2f\n', Ns_v, t90_v);

% Альтернативні параметри
r2 = 0.3; K2 = 150; C0_2 = 0.5; a2 = 0.02; b2 = 0.05; Cmax_2 = 1.5;
N0_2 = 5; C_init_2 = 1.2;

y_verhulst_alt = zeros(num_steps, 2);
y_verhulst_alt(1, :) = [N0_2, C_init_2];
for i = 1:num_steps-1
    y_verhulst_alt(i+1, :) = runge_kutta(@verhulst_rhs, t(i), y_verhulst_alt(i, :)', h, r2, K2, C0_2, a2, b2, Cmax_2)';
end
[Ns_va, t90_va] = compute_metrics(t, y_verhulst_alt(:, 1));
printf('Вергула (альтернативний): N_steady = %.2f, t_90 = %.2f\n', Ns_va, t90_va);

%% МОДЕЛЬ ГОМПЕРТЦА
% Базові
r = 0.5; K = 100; N0 = 10;
y_gomp = zeros(num_steps,1);
y_gomp(1) = N0;
for i = 1:num_steps-1
    y_gomp(i+1) = runge_kutta_scalar(@(t, N) r * max(N, eps) * log(K / max(N, eps)), t(i), y_gomp(i), h);
end
[Ns_g, t90_g] = compute_metrics(t, y_gomp);
printf('Гомпертца (базовий): N_steady = %.2f, t_90 = %.2f\n', Ns_g, t90_g);

% Альтернативні
r2 = 0.3; K2 = 200; N0_2 = 5;
y_gomp_alt = zeros(num_steps,1);
y_gomp_alt(1) = N0_2;
for i = 1:num_steps-1
    y_gomp_alt(i+1) = runge_kutta_scalar(@(t, N) r2 * max(N, eps) * log(K2 / max(N, eps)), t(i), y_gomp_alt(i), h);
end
[Ns_ga, t90_ga] = compute_metrics(t, y_gomp_alt);
printf('Гомпертца (альтернативний): N_steady = %.2f, t_90 = %.2f\n', Ns_ga, t90_ga);

%% МОДЕЛЬ БЕРТАЛАНФІ
% Базові
a = 0.8; b = 0.1; N0 = 10;
y_bert = zeros(num_steps,1);
y_bert(1) = N0;
for i = 1:num_steps-1
    y_bert(i+1) = runge_kutta_scalar(@(t, N) a * max(N, eps)^(2/3) - b * max(N, eps), t(i), y_bert(i), h);
end
[Ns_b, t90_b] = compute_metrics(t, y_bert);
printf('Берталанфі (базовий): N_steady = %.2f, t_90 = %.2f\n', Ns_b, t90_b);

% Альтернативні
a2 = 0.6; b2 = 0.15; N0_2 = 5;
y_bert_alt = zeros(num_steps,1);
y_bert_alt(1) = N0_2;
for i = 1:num_steps-1
    y_bert_alt(i+1) = runge_kutta_scalar(@(t, N) a2 * max(N, eps)^(2/3) - b2 * max(N, eps), t(i), y_bert_alt(i), h);
end
[Ns_ba, t90_ba] = compute_metrics(t, y_bert_alt);
printf('Берталанфі (альтернативний): N_steady = %.2f, t_90 = %.2f\n', Ns_ba, t90_ba);

%% ГРАФІКИ
% Вергула базовий — N(t)
figure;
plot(t, y_verhulst(:,1), 'b-', 'LineWidth', 2);
xlabel('Час'); ylabel('Розмір пухлини N(t)');
title('Модель Вергула');
grid on;
print -dpng 'verhulst_base_N.png';

% Вергула базовий — C(t)
figure;
plot(t, y_verhulst(:,2), 'r-', 'LineWidth', 2);
xlabel('Час'); ylabel('Концентрація поживних речовин C(t)');
title('Модель Вергула');
grid on;
print -dpng 'verhulst_base_C.png';

% Вергула альтернативний — N(t)
figure;
plot(t, y_verhulst_alt(:,1), 'b-', 'LineWidth', 2);
xlabel('Час'); ylabel('Розмір пухлини N(t)');
title('Модель Вергула');
grid on;
print -dpng 'verhulst_alt_N.png';

% Вергула альтернативний — C(t)
figure;
plot(t, y_verhulst_alt(:,2), 'r-', 'LineWidth', 2);
xlabel('Час'); ylabel('Концентрація поживних речовин C(t)');
title('Модель Вергула');
grid on;
print -dpng 'verhulst_alt_C.png';

% Гомпертц базовий — N(t)
figure;
plot(t, y_gomp, 'g-', 'LineWidth', 2);
xlabel('Час'); ylabel('Розмір пухлини N(t)');
title('Модель Гомпертца');
grid on;
print -dpng 'gompertz_base.png';

% Гомпертц альтернативний — N(t)
figure;
plot(t, y_gomp_alt, 'g-', 'LineWidth', 2);
xlabel('Час'); ylabel('Розмір пухлини N(t)');
title('Модель Гомпертца');
grid on;
print -dpng 'gompertz_alt.png';

% Берталанфі базовий — N(t)
figure;
plot(t, y_bert, 'm-', 'LineWidth', 2);
xlabel('Час'); ylabel('Розмір пухлини N(t)');
title('Модель Берталанфі');
grid on;
print -dpng 'bertalanffy_base.png';

% Берталанфі альтернативний — N(t)
figure;
plot(t, y_bert_alt, 'm-', 'LineWidth', 2);
xlabel('Час'); ylabel('Розмір пухлини N(t)');
title('Модель Берталанфі');
grid on;
print -dpng 'bertalanffy_alt.png';

