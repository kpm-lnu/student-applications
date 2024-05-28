clc
clear
close all
syms ae s g ai k p b m L

J=[0        -ae*s -ai*s g 0;
0 ae*s - p - k  ai*s 0 0;
 0 k -b-m 0 0;
  0 p b -g 0;
0 0 m 0 0;];

Pol=(expand(det(J-L.*eye(size(J,1)))))
c=coeffs(Pol,L)
roots=solve(Pol,L)
G=[-c(4) 1 0 0 0;-c(2) -c(3) -c(4) 1 0;0 0 -c(2) -c(3) -c(4);0 0 0 0 -c(2);0 0 0 0 0]
for i=1:size(G,1)
delta(i)=det(G(1:i,1:i));
minors(i)=vpa(subs(delta(i),[ae g ai k p s b m],[0.65 0.001 0.005 0.05 0.08 0.1 0.1 0.02]));
end

try
r2=solve(minors(3),s);
disp(strjoin(["minors - ",string(double(minors))]))
catch
    
end













