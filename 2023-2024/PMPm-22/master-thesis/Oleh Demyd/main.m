global beta gamma Lambda mu theta psi omega 

beta = 0.61229;
gamma = 0.0714;
Lambda = 0.04426;
mu = 0.04426;
omega = 0.1;
theta = 0.01;
psi = 0.6;
    
tstart = 0;
tstop = 100;

S0 = 0.99;
V0 = 0;
I0 = 0.01;

[time, result] = ode45(@sir, [tstart, tstop], [S0, V0, I0]);

susceptible = result(:, 1);
infected = result(:, 3);
vac = result(:,2);
recovered = 1 - result(:, 1) -  result(:,2) - result(:, 3);

R0_vacc = (Lambda * beta * (theta + mu + omega - (psi * omega))) / ...
    (mu * (mu + omega + theta) * (gamma + mu))

hold on;
plot(time, susceptible, '-b', 'LineWidth', 2);
plot(time, infected, '-r', 'LineWidth', 2);
plot(time, recovered, '-g', 'LineWidth', 2);
plot(time, vac, '-m', 'LineWidth', 2);

title(['З вакцинацією']);

legend('Сприйнятливі', ...
 'Інфіковані', 'Одужалі', 'Вакциновані');

xlabel('Час'); ylabel('Пропорція населення'); grid on

beta = 0.61229;
gamma = 0.0714;
Lambda = 0.04426;
mu = 0.04426;
omega = 0;
theta = 0;
psi = 0;

[time, result] = ode45(@sir, [tstart, tstop], [S0, V0, I0]);

susceptible = result(:, 1);
infected = result(:, 3);
vac = result(:,2);
recovered = 1 - result(:, 1) -  result(:,2) - result(:, 3);

R0 = (Lambda * beta * (theta + mu + omega - (psi * omega))) / ...
    (mu * (mu + omega + theta) * (gamma + mu))

figure 

hold on;
plot(time, susceptible, '-b', 'LineWidth', 2);
plot(time, infected, '-r', 'LineWidth', 2);
plot(time, recovered, '-g', 'LineWidth', 2);
plot(time, vac, '-m', 'LineWidth', 2);

title(['Без вакцинації']);

legend('Сприйнятливі', ...
 'Інфіковані', 'Одужалі', 'Вакциновані');

xlabel('Час'); ylabel('Пропорція населення'); grid on
