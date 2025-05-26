clear
tstart = 0;
tstop = 80;

S0 = 0.98;
V0 = 0.01;
I0 = 0.01;

[time, result] = ode45(@sir, [tstart, tstop], [S0, V0, I0]);

susceptible = result(:, 1);
infected = result(:, 3);
vac = result(:,2);
recovered = 1 - result(:, 1) -  result(:,2) - result(:, 3);

hold on;
plot(time, susceptible, '-b');
plot(time, infected, '-r');
plot(time, recovered, '-g');
plot(time, vac, '-m');

title(['З вакцинацією, омега = 0,01']);

legend('Сприйнятливі', ...
 'Інфіковані', 'Одужалі', 'Vacc');

xlabel('Час'); ylabel('Пропорція населення'); grid on