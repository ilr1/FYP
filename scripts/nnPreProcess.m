%we're going to figure out what our system is using a nn
%sampling variables
fs = 1e3;
t = 0:1/fs:1;
trainingSize = 1000;
xTrain = zeros(1,size(t,2),1,trainingSize);
yTrain = zeros(1,size(t,2),1,trainingSize);
count = 1;
randF = 100*rand(1,1000);
randA = 0.5 + rand(1,1000);
for i = randperm(trainingSize,trainingSize)
    f = randF(i);
    x = randA(i)*cos(2*pi*f*t);
    yin(1,:,1) =  x;
    y = predict(netTF, yin);
    yTrain(1,:,1, count) = x;
    xTrain(1,:,1, count) = y;
    count = count + 1;
end

yTrain = permute(yTrain,[1 3 2 4]);

plot(permute(yTrain(1,1,:,1),[1 3 2 4]))
%%
layers = [
    imageInputLayer([1 length(t) 1],'name','input')...
    fullyConnectedLayer(length(t),'name', 'hl1')...
    regressionLayer('name','output')
];

lgraph = layerGraph(layers);
plot(lgraph);
%%
options = trainingOptions('sgdm', ...
'LearnRateSchedule','piecewise', ...
'LearnRateDropFactor',0.2, ...
'LearnRateDropPeriod',5, ...
'MaxEpochs',20, ...
'MiniBatchSize',64, ...
'Plots','training-progress');

netInv = trainNetwork(xTrain,yTrain,layers,options);
%%
%netTest = netTF(xTrain(1,:,1,1));
close all; clear yin;
yin(1,:,1) =  cos(2*pi*10*t + pi)+1;
yout = predict(netTF,yin);
%plot(t,yin);
%hold on
plot(t,passSignalThroughTF(yin,t));
hold on
plot(t, predict(netTF, yin));

%legend('yin','throughTF','throughNet');
legend('throughTF','throughNet');