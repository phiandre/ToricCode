clear
clc
aa=1
bb=100000

finalReward = 10;

x=linspace(aa,bb);
w=pi/70000;
b=-50000;
A=.5*finalReward;
B=.5*finalReward;
y=A*tanh(w*(x+b))+B;
plot(x,y)
grid on