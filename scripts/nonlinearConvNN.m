%we're going to figure out what our system is using a nn
%sampling variables
fs = 1e3;
t = 0:1/fs:1;
trainingSize = 2000;
xTrain = zeros(1,size(t,2),1,trainingSize);
yTrain = zeros(1,size(t,2),1,trainingSize);
count = 1;
randF = 100*rand(1,trainingSize)+0.1;
randF2 = 100*rand(1,trainingSize)+0.1;
randA = 0.5 + rand(1,trainingSize);
randB = rand(1,trainingSize);
for i = randperm(trainingSize,trainingSize)
    %f = randF(i);
    f = 100*rand(1,100)+0.1;
    f1 = randF(i);
    
    %x = randA(i)*cos(2*pi*f*t) + randB(i)*cos(2*pi*f1*t);
    %x = rand(size(t));
    %x = wgn(1,length(t), 1, 1,i);
    
    %f = mod(i,100)+1;
    x = 0;
    for j = 1:100
        x = x + cos(2*pi*f(j)*t);
    end
    
    y = passSignalThrough_NL_TF(x,t);
    xTrain(1,:,1, count) = x;
    yTrain(1,:,1, count) = y';
    count = count + 1;
end

yTrain = permute(yTrain,[1 3 2 4]);


XTrain = xTrain;
YTrain = yTrain;

idx = randperm(size(XTrain,4),trainingSize/5);
XValidation = XTrain(:,:,:,idx);
XTrain(:,:,:,idx) = [];
YValidation = YTrain(idx);
%YTrain(idx) = [];


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
   convolution2dLayer([1 250], 20, 'Padding', 'same', 'Name', 'CONV')...
   batchNormalizationLayer('Name','bn')...
   reluLayer('Name','Relu1')...
   ...%maxPooling2dLayer([1 floor(length(t)/10)],'Name','mp2d')...
   fullyConnectedLayer(length(t),'name', 'hl1')...
   reluLayer('Name','Relu2')...
   fullyConnectedLayer(length(t),'name', 'hl2')...
   regressionLayer('name','output')
];
%
%%
lgraph = layerGraph(layers);
plot(lgraph);

options = trainingOptions('adam', ...
'LearnRateSchedule','piecewise', ...
'InitialLearnRate',0.01,...
'LearnRateDropFactor',0.1, ...
'LearnRateDropPeriod',5, ...
'MaxEpochs',30, ...
'MiniBatchSize',64, ...
'Shuffle','once',...
...%'ValidationData',{XValidation,YValidation}, ...
...%'ValidationFrequency',30, ...
'Plots','training-progress');
%%
netTF = trainNetwork(xTrain,yTrain,layers,options);
%%
%netTest = netTF(xTrain(1,:,1,1));
%close all; 
clear yin;
yin(1,:,1) =  10*sin(2*pi*20*t);% + 1*cos(2*pi*75*t);%+ 0.5*cos(2*pi*71*t);
yout = predict(netTF,yin);
%plot(t,yin);
%hold on
plot(t,passSignalThrough_NL_TF(yin,t));
hold on
plot(t, predict(netTF, yin));
%legend('yin','throughTF','throughNet');
legend('throughTF','throughNet');

figure
plot(abs(fft(passSignalThrough_NL_TF(yin,t))));
hold on
plot(abs(fft(predict(netTF, yin))));
legend('throughTF','throughNet');
