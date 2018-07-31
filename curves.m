clear
clc
% Viktigt! För representativa kurvor, ange antal iterationer
iterations = 3600*8;

% För konstant form oavsett antal iterationer, sätt värde för kurvan till 1
epsilonShape = 1;
errorShape = 0;
groundStateShape = 0;


% 
%       Epsilon Decay
%

subplot(2,2,1)




alfa = -0.9; % "flyttar änden" på kurvan

x = 1:iterations;

if epsilonShape ==1
    k = iterations/10; % "flyttar mitten" på kurvan
else
    k = 20000;
end
y = ((k+x)/k).^alfa;


plot(x,y)
grid on

title('Esilon Decay')

axis([1 iterations 0 1])

% 
%       Error Rate Growth
%

subplot(2,2,2)
aa=1;
bb=iterations;

finalProbability = .1;
initialProbability = .04;

x=linspace(aa,bb);

if errorShape ==1
    w=pi/(0.125*iterations);
    b=-0.13*iterations;
else
    w=pi/25000;
    b=-26000;
end
    
A=.5*finalProbability-.5*initialProbability;
B=.5*finalProbability+.5*initialProbability;
p=A*tanh(w*(x+b))+B;
plot(x,p)
grid on
title('Ground State Reward')

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
if groundStateShape ==1
    w=pi/(0.275*iterations);
    b=-0.39*iterations;
else
    w=pi/55000;
    b=-78000;
end
A=.5*finalReward;
B=.5*finalReward;
y=A*tanh(w*(x+b))+B;
plot(x,y)
grid on
title('Ground State Reward')