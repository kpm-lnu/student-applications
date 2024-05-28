function dy = unaimed_fire5(t, y, beta, rho)
    dy = zeros(2,1);
    dy(1) = -rho * y(1);
    dy(2) = -beta * y(2);
end