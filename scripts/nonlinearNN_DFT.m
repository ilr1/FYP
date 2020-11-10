%we're going to figure out what our system is using a nn
%sampling variables
fs = 1e3;
t = 0:1/fs:1;
trainingSize = 10000;
xTrain = zeros(1,size(t,2),1,trainingSize);
yTrain = zeros(1,size(t,2),1,trainingSize);
count = 1;
randF = 100*rand(1,trainingSize)+0.1;
randF2 = 100*rand(1,trainingSize)+0.1;
randA = 0.5 + rand(1,trainingSize);
randB = rand(1,trainingSize);
for i = randperm(trainingSize,trainingSize)
    f = randF(i);
    f1 = randF(i);
    
    x = randA(i)*cos(2*pi*f*t) + randB(i)*cos(2*pi*f1*t);
    %x = rand(size(t));
    %x = wgn(1,length(t), 10, 1,i);
    
    %f = mod(i,100)+1;
    %x = cos(2*pi*f*t);
    y = passSignalThrough_NL_TF(x,t);
    xTrain(1,:,1, count) = abs(fft(x));
    yTrain(1,:,1, count) = abs(fft(y'));
    count = count + 1;
end

yTrain = permute(yTrain,[1 3 2 4]);
%% create layers of series network
clear layers;
% layers = [
%     imageInputLayer([1 length(t) 1],'name','input')...
%     fullyConnectedLayer((length(t)-1)/10,'name', 'hl1')...
%     %reluLayer('Name','Relu')...
%     %fullyConnectedLayer(length(t),'name', 'hl2')...
%     regressionLayer('name','output')
% ];

layers = [
   imageInputLayer([1 length(t) 1],'name','input')...
   fullyConnectedLayer(length(t),'name', 'hl0')...
   reluLayer('Name','Relu1')...
   fullyConnectedLayer(length(t),'name', 'hl1')...
   reluLayer('Name','Relu2')...
   fullyConnectedLayer(length(t),'name', 'hl2')...
   reluLayer('name','Relu3')...
   fullyConnectedLayer(length(t),'name', 'hl3')...
   regressionLayer('name','output')
];
%
%%
lgraph = layerGraph(layers);
plot(lgraph);

options = trainingOptions('adam', ...
'LearnRateSchedule','piecewise', ...
'LearnRateDropFactor',0.5, ...
'LearnRateDropPeriod',3, ...
'MaxEpochs',10, ...
'MiniBatchSize',64, ...
'Plots','training-progress');
%%
netTF = trainNetwork(xTrain,yTrain,layers,options);
%%
%netTest = netTF(xTrain(1,:,1,1));
close all; clear yin;
y =  1*sin(2*pi*16*t) + 1*cos(2*pi*51*t);
yin(1,:,1) = abs(fft(y));
yout = predict(netTF,yin);
%plot(t,yin);
%hold on
plot(t,passSignalThrough_NL_TF(y,t));
hold on
plot(t, ifft(predict(netTF, yin)));
%legend('yin','throughTF','throughNet');
legend('throughTF','throughNet');

figure
plot(abs(fft(passSignalThrough_NL_TF(y,t))));
hold on
plot(predict(netTF, yin));
legend('throughTF','throughNet');
