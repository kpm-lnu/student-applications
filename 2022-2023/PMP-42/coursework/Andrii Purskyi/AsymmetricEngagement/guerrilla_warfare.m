function dy = guerrilla_warfare(t, y, beta, rho, R0)
    dy = zeros(2,1);
    dy(1) = max(-rho * y(2), -y(1));
    dy(2) = max(-beta * y(1) * (y(2)/R0), -y(2));
end