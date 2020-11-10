syms y(t)
[V] = odeToVectorField(diff(y, 2) + 0.1*diff(y) + y == sin(t))

M = matlabFunction(V,'vars', {'t','Y'})

sol = ode45(M,[0 20],[2 0]);

fplot(@(x)deval(sol,x,1), [0, 20])