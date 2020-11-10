close all; clc;
%% we're gonna gonna run a training loop on this and plug the new value into
%%the net for the next run


%Duration of signal
s = 20;
%Read in 10 seconds of music and global vars
Fs = 44100;
samples = [1,s*Fs];
filePath = "Ludwig_van_Beethoven_-_Concerto_for_Piano_and_Orchestra_no._4_in_G_major_-_3._Rondo_(Wilhelm_Kempff,_Ferdinand_Leitner,_Berliner_Philharmoniker,_1962).flac";
[signal, Fs] = audioread(filePath, samples);
d = daq("ni");
prevSignal = signal(:,1);
l = length(prevSignal);
%input
X = tonndata(prevSignal,false,false);


%trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
%inputDelays = 1:2;
%feedbackDelays = 1:2;
%hiddenLayerSize = 10;
%net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);
%closeloop(net);
numFeatures = 12;
numHiddenUnits = 125;
numResponses = 1;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 100;
miniBatchSize = 27;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Data reading
%output object
data = [];

%Add Channels
%Add channels and set channel properties, if any.
addoutput(d,"Dev1","ao0","Voltage");
chanInput = addinput(d,"Dev1","ai0","Voltage");
chanInput.Range = [-10,10];
%Set DataAcquisition Rate
%Set scan rate.
d.Rate = Fs;
amplitudePeakToPeak_ch1 = 2;

%% Training loop
for i = 1:5
outputSignal = prevSignal;
flush(d);
%Generate Signal Data
%Write the signal data.
preload(d,outputSignal);
start(d,"repeatoutput");
pause(1); % Adjust the duration of signal generation.

%Read Data
%Read the data in timetable format.
data = read(d,seconds(s)).Variables;

stop(d);

%Clean Up
%Clear all DataAcquisition and channel objects.
clear sound;

% now for nn stuff
%output
net = trainNetwork(prevSignal, data, layers, options);

T = tonndata(data(1:l),false,false);
[x,xi,ai,t] = preparets(net,X,{},T);

net = trainNetwork(X, T,layers,options);
% % [net,tr] = train(net,x,t,xi,ai,'useGPU','yes');
prevSignal = cell2mat(net(x,xi,ai))';


end

clear d
%%
filename = 'beethovenData.mat';
%save(filename,'data','outputSignal')