close all; clc;
%% we're gonna gonna run a training loop on this and plug the new value into
%%the net for the next run


%Duration of signal
%s = 10;
%Read in 10 seconds of music and global vars
%Fs = 44100;
%samples = [1,s*Fs];
%filePath = "Ludwig_van_Beethoven_-_Concerto_for_Piano_and_Orchestra_no._4_in_G_major_-_3._Rondo_(Wilhelm_Kempff,_Ferdinand_Leitner,_Berliner_Philharmoniker,_1962).flac";
%[signal, Fs] = audioread(filePath, samples);
%prevSignal = signal(:,1);
%l = length(prevSignal);
%input
%X = tonndata(prevSignal,false,false);

layers = [ ...
    sequenceInputLayer(10000)
    flattenLayer
    fullyConnectedLayer(10000)
    %regressionLayer
    acousticLayer("Speaker_Mic")];

maxEpochs = 1;

options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',5, ...
    'MaxEpochs',maxEpochs, ...
    'Plots','training-progress');

%% Training loop
for i = 1:5

% now for nn stuff
%output
net = trainNetwork(data, layers, options);

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
