%syms y(t)
F_0 = 1;
m = 1;
chi = 1;
omega_0 = 
ode = diff(diff(x,t),t) + 2*chi*omega_0*diff(x,t) + omega_0*omega_0*x == (F_0/m)*sin(t);

tspan = [0 5];
y0 = 0;
[t,y] = ode23(@(t,y) 2*t, tspan, y0);