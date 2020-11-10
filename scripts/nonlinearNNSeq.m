%we're going to figure out what our system is using a nn
%sampling variables
fs = 1e3;
t = 0:1/fs:1;
trainingSize = 1000;
xTrain = zeros(1,size(t,2),1,trainingSize);
yTrain = zeros(1,size(t,2),1,trainingSize);
count = 1;
randF = 100*(rand(1,trainingSize)+0.01);
randF2 = 100*(rand(1,trainingSize)+0.01);
randA = 0.5 + rand(1,trainingSize);
randB = rand(1,trainingSize);

for i = randperm(trainingSize,trainingSize)
    f = mod(i,100)+1;
    %f = randF(i);
    f1 = randF(i);
    
    %x = randA(i)*cos(2*pi*f*t);% + randB(i)*cos(2*pi*f1*t);
    x = cos(2*pi*f*t);% + randB(i)*cos(2*pi*f1*t);
    
    
    %x = rand(size(t));
    %x = wgn(1,length(t), 10, 1,i);
    
    %f = mod(i,100)+1;
    %x = cos(2*pi*f*t);
    %y = passSignalThrough_NL_TF(x,t);
    y = passSignalThroughTF(x,t);
    xTrain(1,:,1, count) = x;
    yTrain(1,:,1, count) = y';
    count = count + 1;
end

yTrain = permute(yTrain,[1 3 2 4]);
%%
XTrain = squeeze(xTrain);
YTrain = squeeze(yTrain);
%% create layers of series network
clear layers;
% layers = [
%     imageInputLayer([1 length(t) 1],'name','input')...
%     fullyConnectedLayer((length(t)-1)/10,'name', 'hl1')...
%     %reluLayer('Name','Relu')...
%     %fullyConnectedLayer(length(t),'name', 'hl2')...
%     regressionLayer('name','output')
% ];

inputSize = length(t);
numHiddenUnits1 = 20;
numHiddenUnits = 50;
numClasses = length(t);

layers = [ ...
    sequenceInputLayer(inputSize,'Name','Input Sequence Layer')
    %lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    %dropoutLayer(0.2)
    lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','LSTM Layer')
    %bilstmLayer(numHiddenUnits)
    %fullyConnectedLayer(200)
    %reluLayer()
    %fullyConnectedLayer(50)
    %dropoutLayer(0.5)
    fullyConnectedLayer(numClasses,'Name','FC sequence Layer')
    regressionLayer('Name','Output Layer')];
%
%
lgraph = layerGraph(layers);
plot(lgraph);
title('Basic RNN network');
legend('Layers');
%%
options = trainingOptions('adam', ...
'GradientThreshold',1, ...
'LearnRateSchedule','piecewise', ...
'InitialLearnRate', 1e-1,...
'LearnRateDropFactor',0.2, ...
'LearnRateDropPeriod',3, ...
'MaxEpochs',50, ...
'MiniBatchSize',32, ...
'Shuffle', 'every-epoch',...
'Plots','training-progress');
%%
netTF = trainNetwork(XTrain,YTrain,layers,options);
%%
%netTest = netTF(xTrain(1,:,1,1));
close all; clear yin;
yin(1,:,1) =  10*cos(2*pi*80*t) + 10*cos(2*pi*31*t);

yin = squeeze(yin)';

netTF = resetState(netTF);
[netTF, yout] = predictAndUpdateState(netTF,XTrain);

%yout = predict(netTF,yin,'MiniBatchSize',1);

% X = yin;
% numTimeSteps = size(X,2);
% for i = 1:numTimeSteps
%     v = X(:,i);
%     [net,score] = predictAndUpdateState(net,v);
%     scores(:,i) = score;
% end

%%
plot(t,yin,t,passSignalThroughTF(yin,t));
hold on
plot(t, yout);
title('Time domain comparison of System Response vs RNN output');
xlabel('Time(s)');
ylabel('Amplitude (arbitrary)');
legend('throughTF','throughNet');
%%
figure
L = length(t);
Y = fft(passSignalThroughTF(yin,t));
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
plot(f(1:200),P1(1:200)) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
%plot(abs());
%%
hold on
Y = fft(predict(netTF, yin));
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
plot(f(1:200),P1(1:200)) 
legend('throughTF','throughNet');

%%
plot(abs(fft(predict(netTF, yin))));
title('Frequency domain comparison of System Response vs RNN output');
xlabel('Freq(s)');
ylabel('Amplitude (arbitrary)');
legend('throughTF','throughNet');
