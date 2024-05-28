clear
tstart = 0;
tstop = 70;

S0 = 0.99;
V0 = 0;
I0 = 0.01;

[time, result] = ode45(@sir, [tstart, tstop], [S0, V0, I0]);

susceptible = result(:, 1);
infected = result(:, 3);
vac = result(:,2);
recovered = 1 - result(:, 1) -  result(:,2) - result(:, 3);

hold on;
plot(time, susceptible, '-b','LineWidth',2);
plot(time, infected, '-r','LineWidth',2);
plot(time, recovered, '-g','LineWidth',2);
plot(time, vac, '-m','LineWidth',2);

title(['З вакцинацією, псі = 0,6']);

legend('Сприйнятливі', ...
 'Інфіковані', 'Одужалі', 'Vacc');

xlabel('Час'); ylabel('Пропорція населення'); grid on