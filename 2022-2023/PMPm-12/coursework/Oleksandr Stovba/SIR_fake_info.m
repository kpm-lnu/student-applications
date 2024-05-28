clc
%clear
close all
%2021 Ukraine
ii=0.676;
h=0.779;
% %2021 Russia
% ii=0.9;
% h=0.01;
dydt = @(t,y,a,b)[-b*y(1).*y(2);
                   b*y(1).*y(2)-a*y(2);%SIR
                   a*y(2);];
                 
a=h/100;b=ii/10;%param
tspan = [0 1000];%time
y0= [0.9999  0.0001 0 ];%initial val
n=tspan(end)+1;
sol = ode45(@(t,y) dydt(t,y,a,b), tspan, y0);%Runge-Kutta num. solution
t=linspace(tspan(1),tspan(end),n)';%часовий інтервал
y=deval(sol,t)';%розвязок на часовому інтервалі
plot(t,y(:,1),t,y(:,2),t,y(:,3),'LineWidth',1.4)%графік розвязку
legend('possibility of fake bealiving','fast fake spread','inactive to spread fake')
title('Ukraine')
figure
plot3(y(:,1),y(:,2),y(:,3),'b',y(1,1),y(1,2),y(1,3),'go',y(end,1),y(end,2),y(end,3),'ro','LineWidth',2.4)%графік розвязку
legend('phase trajectory','start','end')
title('Ukraine')

%Аналіз жорсткості
J=@(a,b,S0,I0)[-b*I0 -b*S0 0;b*I0 b*S0-a 0;0 a 0];%Якобіан

J_Ukr=J(a,b,y(1),y(2));

lambda=eig(J_Ukr);%потрібне подальше дослідження на стійкість
bool=lambda==0;
for i=1:length(lambda)
   if bool(i)==1
      k=i;
     lambda(k)=[];
     sig=abs(max(real(lambda)))./abs((min(real(lambda)))) %число жорсткості
   end
    
    
end

disp("Оскільки число жорсткості = 1,а також по графіку можна сказати що довіри до фейкових новин немає взагалі,тобто синій графік спадає.")
disp("Також 200 день-це день максимального розповсюдження фейкових новин,тим не менше довіра все одно йде на спад а разом з нею і фейкове розповсюдження ")
disp("Натомість росте стійкість та несприйнятливість новин на протязі 1000 днів")
disp("В загальному,якщо би дійсно параметри i та h не мінялись проягом усіх цих днів то Україну можна вважати стійкою до фейкових новин")

















