close all;
[X,T] = simpleseries_dataset;
Xnew = X(81:100);
X = X(1:80);
T = T(1:80);

net = timedelaynet(1:2,10);
[Xs,Xi,Ai,Ts] = preparets(net,X,T);
net = train(net,Xs,Ts,Xi,Ai);
view(net)

[Y,Xf,Af] = net(Xs,Xi,Ai);
perf = perform(net,Ts,Y);

[netc,Xic,Aic] = closeloop(net,Xf,Af);
view(netc)

y2 = netc(Xnew,Xic,Aic);
