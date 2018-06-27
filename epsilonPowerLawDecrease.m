clear
clc
alfa = -1;

x = 1:100000;

k = 12000;

y = ((k+x)/k).^alfa;


plot(x,y)
grid on

axis([x(1) x(length(x)) 0 1])