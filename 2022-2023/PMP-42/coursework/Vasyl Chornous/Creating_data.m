clc
clear
close all

C=readcell('stat_ger.xlsx');

 
 
ij=double(string((C(3:end,2))));
Alli=sum(ij);
ij=ij./Alli;

dj=double(string((C(3:end,3))));
Alld=sum(dj);
dj=dj./Alld;
% dj=string(C(1:end,2));
% dj=fillmissing(dj,'constant',"0(");
% pat =digitsPattern+("(");
% dj=extract(dj,pat);
% dj=double(strrep(dj,"(",""))./All;

t=1:length(dj);
y0= [(1-1/Alld-1/Alli) 1/Alld 1/Alli 0 0];
b0=[0.05 0.1 0.02];
lb=[0 0 0] ;
ub=[1 .1 .1 ];
%[b_opt_grad,J_val_grad]=fmincon(@(b)Jlsq(b,ij,dj,y0),b0,[],[],[],[],lb,ub)
[b_opt,J_val]=patternsearch(@(b)Jlsq(b,t,ij,dj,y0),b0,[],[],[],[],lb,ub)
%[b_opt,J_val]=ga(@(b)Jlsq(b,t,ij,dj,y0),length(b0),[],[],[],[],lb,ub);

[~,sol,y]=Jlsq(b_opt,t,ij,dj,y0);
y=y';
figure

plot(t,ij,t,y(3,:))
ylabel('|');
xlabel('Дні');
legend('experiment data','mdl data')
figure

plot(t,dj,t,y(5,:))
ylabel('d');
xlabel('Дні');
legend('experiment data','mdl data')










