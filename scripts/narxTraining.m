pos = 2;
singleI = iData(:,pos);
singleO = oData(:,pos);


%[x,t] = maglev_dataset;
x = num2cell(singleI.');
t = num2cell(singleO.');
%%
% We can view the sizes of inputs X and targets T.
%
% Note that both X and T have 4001 columns. These represent 4001 timesteps
% of the control current and magnet position.

size(x)
size(t)

%% Time Series Modelling with a Neural Network
% The next step is to create a neural network that will learn to model how
% the magnet changes position.
%
% Since the neural network starts with random initial weights, the results
% of this example will differ slightly every time it is run. The random
% seed is set to avoid this randomness. However this is not necessary for
% your own applications.
%%
setdemorandstream(491218381)

net = narxnet(1:2,1:2,10);
view(net)

%for i = 1
%pos = i;
singleI = iData(:,pos);
singleO = oData(:,pos);

%[x,t] = maglev_dataset;
x = num2cell(singleI.');
t = num2cell(singleO.');

[Xs,Xi,Ai,Ts] = preparets(net,x,{},t);
[net,tr] = train(net,Xs,Ts,Xi,Ai);
%end
%nntraintool
%%
nntraintool('close')

plotperform(tr)
Y = net(Xs,Xi,Ai);

perf = mse(net,Ts,Y)

plotresponse(Ts,Y)
E = gsubtract(Ts,Y);

ploterrcorr(E)

%plotinerrcorr(Xs,E)

%%
net2 = closeloop(net);
view(net2)

[Xs,Xi,Ai,Ts] = preparets(net2,x,{},t);
Y = net2(Xs,Xi,Ai);
plotresponse(Ts,Y)

net3 = removedelay(net);
view(net3)
[Xs,Xi,Ai,Ts] = preparets(net3,x,{},t);
Y = net3(Xs,Xi,Ai);
plotresponse(Ts,Y)
