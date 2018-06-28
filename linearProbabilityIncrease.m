clear
a = 1;
b = 100000;
x = a:b;

pa = 0.02;
pb = 0.10;

p = (pb-pa)/(b-a)*x+pa;

plot(x,p)
grid on