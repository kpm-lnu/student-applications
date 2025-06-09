% Define the parameters
a = 0;  % coefficient a
b = 0.2;  % coefficient b
c = 0;  % coefficient c
d = 0.3;  % coefficient d
u = 0;  % external input u
v = 0;  % external input v

% Define the function that represents the system of differential equations
dydt = @(t, y) [-a*y(1) - b*y(2) + u; -c*y(2) - d*y(1) + v];

% Define the initial conditions
y0 = [300; 200];  % initial values of N1 and N2
tic;
% Set the initial time and empty arrays for time and solutions
t = 0;
time = [];
solution = [];

% Solve the system of differential equations
while y0(1) > 0.01 && y0(2) > 0.01
    % Solve the differential equations for a small time span
    tspan = [t, t+1];  % time span for each iteration
    [t_temp, y_temp] = ode45(dydt, tspan, y0);

    % Append the time and solutions to the main arrays
    time = [time; t_temp];
    solution = [solution; y_temp];

    % Update the initial conditions and time for the next iteration
    y0 = y_temp(end, :);
    t = t_temp(end);
end
elapsed_time = toc;
fprintf('Час виконання алгоритму: %.2f секунд\n', elapsed_time);
% Plot the solutions
figure;
plot(time, solution(:, 1), 'b-', 'LineWidth', 2);
hold on;
plot(time, solution(:, 2), 'r-', 'LineWidth', 2);
hold off;

% Add labels and title
xlabel('Час');
ylabel('Чисельність');
%title('Модель без резерву та контролю');
legend('y1(t)', 'y2(t)');
