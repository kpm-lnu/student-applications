clc
clear
close all


b=[0.65 0.005 0.05 0.08 0.1 0.02 0.001];

tspan = [0 150];
y0= [0.9998 0.0001 0.0001 0 0];

sol = ode45(@(t,y) seir(t,y,b), tspan, y0);
t=sol.x';
y=sol.y';

plot(t,y(:,1),t,y(:,2),t,y(:,3),t,y(:,4),t,y(:,5),'LineWidth',1.4)%графік розвязку
legend('s','e','i','r','d')
title('Графік розвязку SEIR системи при 15 денному карантині')
ylabel('Частка населення відповідної категорії');
xlabel('Дні');



function val=seir(t,y,b)
var3 = fun4var3(t);
val=[-var3*y(1).*y(2)-b(2)*y(1).*y(3)+b(7)*y(4);
                     var3*y(1).*y(2)+b(2)*y(1).*y(3)-b(3)*y(2)-b(4)*y(2);
                     b(3)*y(2)-b(5)*y(3)-b(6)*y(3);
                     b(5)*y(3)+b(4)*y(2)-b(7)*y(4);
                     b(6)*y(3)];
end


function var3 = fun4var3(t)
  if t <=15 || t>30
    var3 = 0.65;
  else
    var3 = 0.1;
  end
end






