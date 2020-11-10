params = load("C:\Isaac\Documents\University\FYP\Code\params_2020_09_23__11_25_58.mat");
layers = [
    imageInputLayer([10000 1 1],"Name","imageinput")
    fullyConnectedLayer(1000,"Name","fc_1")
    reluLayer("Name","relu_1")
    fullyConnectedLayer(10000,"Name","fc_2")
    reluLayer("Name","relu_2")
    regressionLayer("Name","regressionoutput")];
plot(layerGraph(layers));
dd = open('From50HzTo1kHz.mat')
fs = dd.Fs;
t = 0:1/fs:1-1/fs;

freqs = dd.freqA;
input = zeros(10000, length(freqs));
for i = 1:length(freqs)
    input(:,i) = sin(2*pi*freqs(i)*t);
end

output = dd.data;

for i = 1:length(freqs)
    input(:,i) = input(:,i)/max(input(:,i));
    output(:,i) = output(:,i)/max(output(:,i));
end
%plot(input,output)

preShuff = [input(5001:end,:) output(5001:end,:)];

shuff = preShuff(:,randperm(size(preShuff', 1)));

input = shuff(:,1:length(freqs));
output = shuff(:,length(freqs)+1:2*length(freqs));

figure
plot(t,input(:,1));
xlim([-0 1])
ylim([-1 1])
hold on
plot(t,output(:,1));
xlim([-0 1])
ylim([-1 1])
hold off

filterSize = 5;
numHiddenUnits = 200;

layers = [
    sequenceInputLayer(5000,"Name","imageinput")
    
    %sequenceFoldingLayer("Name","fold")
    %flattenLayer("Name","flat")

    %lstmLayer(1000,"Name","BLSTM")
    %convolution2dLayer(filterSize,numFilters,'Name','conv')
    %batchNormalizationLayer('Name','bn')
    %reluLayer('Name','relu')
    
    %sequenceUnfoldingLayer('Name','unfold')
    %flattenLayer('Name','flatten')
    
    lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','lstm')

    %lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','lstm')

    fullyConnectedLayer(5000,"Name","fc_2")
    
    %reluLayer("Name","relu_2")
    regressionLayer("Name","regressionoutput")];
plot(layerGraph(layers));
options = trainingOptions('rmsprop', ...
'MiniBatchSize',miniBatchSize, ...
'MaxEpochs',100, ...
'InitialLearnRate',1e-2, ...
'LearnRateSchedule','piecewise', ...
'LearnRateDropFactor',0.1, ...
'LearnRateDropPeriod',40, ...
'Shuffle','every-epoch', ...
'ValidationFrequency',validationFrequency, ...
'Plots','training-progress', ...
'Verbose',false);
net = trainNetwork(mat2cell(input,5000,951),mat2cell(output,5000,951),layers,options);
inputed = dd.oData(:,4);
inputed = 0.75*inputed;
outputed = dd.iData(:,4);

Y = predict(net, inputed);

outputed = outputed/max(outputed);
%plot(inputed) 

plot(Y)
hold on
plot(outputed)
legend
figure
plot(outputed,Y,'.')
xlim([-1 1])
ylim([-1 1])

figure
plot(abs(fft(Y)))

figure
plot(abs(fft(outputed)))


