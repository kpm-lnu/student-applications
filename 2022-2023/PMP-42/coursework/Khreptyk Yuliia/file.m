clc
clear
close all
m = 0;
tspan=[0,2];
xspan=[0,1];
N=25;%Сітка [x_0 x_{end}]x[t(0) t_{end}]
x = linspace(xspan(1),xspan(end),N);%дискретизуємо просторову змінну х
t = linspace(tspan(1),tspan(end),N);%дискретизуємо часову змінну t
D=0.2;%ex1,ex2
d=10;%ex1,ex2
%c0=1%ex1
c0=0;%ex3
%c0=mean(xspan);%ex2
%ce=1;%ex1
%ce=1000;%ex2
ce(1)=0;
ce(2)=1;%ex3
%alpha=1;%ex1
%alpha=0.001;%ex2
alpha=-1;
sol = pdepe(m,@(x,t,u,DuDx)pdex1pde(x,t,u,DuDx,D,d),@(x)pdex1ic(x,c0),@(xl,ul,xr,ur,t)pdex1bc(xl,ul,xr,ur,t,D,alpha,ce),x,t);
% Extract the first solution component as u.
y = sol(:,:,1);%наближений розвязок

%виводим графіки
surf(x,t,y) 
title('Numerical solution of 1D reaction-diffusion eqn computed with 25 mesh points.')
xlabel('Distance x')
ylabel('Time t')


figure
plot(x,y(end,:))
title("Solution at t = t_e = "+string(tspan(end)))
xlabel('Distance x')
ylabel('y(x,t_e)')
figure
plot(t,y(:,round(mean([1,N]))))
title("Solution at x =((x_0+x_{end})/2) = "+string((mean(xspan))))
xlabel('Time t')
ylabel('y((x_{end}+x_0)/2,t)')



function [c,f,s] = pdex1pde(x,t,u,DuDx,D,d)
%вигляд нашого рівняння в част похідних
 c = 1;
 f = D*DuDx;
 s =d*u;
end
% --------------------------------------------------------------
function u0 = pdex1ic(x,c0)
%початкова умова по часу
%u0 =3*exp(-5*(x-c0/2).^2) ;%ex2
u0=c0;%ex1
end
% --------------------------------------------------------------
function [pl,ql,pr,qr] = pdex1bc(xl,ul,xr,ur,t,D,alpha,ce)
%крайові умови Неймана
pl = -alpha*(ul-ce(1));
ql = 0;
pr = -alpha*(ur-ce(2));
qr = 0;
end