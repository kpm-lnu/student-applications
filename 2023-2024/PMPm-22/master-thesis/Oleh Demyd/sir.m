function result = sir(t, y)
    global beta gamma Lambda mu theta psi omega 

    result = [Lambda - (omega + mu)*y(1) + theta*y(2) - beta*y(1)*y(3) omega*y(1) - (1-psi)*beta*y(3)*y(2) - ...
     (theta+mu)*y(2) beta*y(1)*y(3) + (1-psi)*(beta*y(2)*y(3)) - gamma*y(3) - mu*y(3)];
    result = result';
end