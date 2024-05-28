function result = sir(t, y)
    beta = 0.61229;
    gamma = 0.0714;
    Lambda = 0.04426;
    mu = 0.04426;
    omega = 0.1;
    theta = 0.01;
    psi = 0.6;
    result = [Lambda - (omega + mu)*y(1) + theta*y(2) - beta*y(1)*y(3) omega*y(1) - (1-psi)*beta*y(3)*y(2) - (theta+mu)*y(2) beta*y(1)*y(3) + (1-psi)*(beta*y(2)*y(3)) - gamma*y(3) - mu*y(3)];
    result = result';
end