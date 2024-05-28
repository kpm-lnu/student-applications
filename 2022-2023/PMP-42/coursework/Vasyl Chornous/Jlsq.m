function [fi,sol,y]=Jlsq(b,t,ij,dj,y0)
n=length(ij);
dydt = @(t,y,b,g)[-0.65*y(1).*y(2)-0.005*y(1).*y(3)+g*y(4);
                     0.65*y(1).*y(2)+0.005*y(1).*y(3)-b(1)*y(2)-0.08*y(2);
                     b(1)*y(2)-b(2)*y(3)-b(3)*y(3);
                     b(2)*y(3)+0.08*y(2)-g*y(4);
                     b(3)*y(3)];
%m=0.02;
g=0.001;
tspan = [t(1) t(end)];
%y0= [0.9998 0.0001 0.0001 0 0];
sol = ode45(@(t,y) dydt(t,y,b,g), tspan, y0);
%t=linspace(tspan(1),tspan(end),n)';%часовий інтервал
y=deval(sol,t)';
fi=sum((y(:,3)-ij).^2+(y(:,5)-dj).^2)/n;
end
