clear
clc
% Viktigt! För representativa kurvor, ange antal iterationer
iterations = 200000;

% 
%       Epsilon Decay
%

subplot(2,2,1)




alfa = -0.8; % "flyttar änden" på kurvan

x = 1:iterations;

k = 20000; % "flyttar mitten" på kurvan

y = ((k+x)/k).^alfa;


plot(x,y)
grid on

title('Esilon Decay')

axis([1 iterations 0 1])

% 
%       Error Rate Growth
%

subplot(2,2,2)



a = 1;
b = iterations;
x = a:b;

pa = 0.02;
pb = 0.10;

p = (pb-pa)/(b-a)*x+pa;

plot(x,p)
grid on
title('Error rate')

% 
%       Ground State reward
%

subplot(2,2,3)
aa=1;
bb=iterations;

finalReward = 5;

x=linspace(aa,bb);
w=pi/90000;
b=-78000;
A=.5*finalReward;
B=.5*finalReward;
y=A*tanh(w*(x+b))+B;
plot(x,y)
grid on
title('Ground State Reward')