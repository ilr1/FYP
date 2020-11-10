close all; clc;
%% we're gonna gonna run a training loop on this and plug the new value into
%%the net for the next run


trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
inputDelays = 1:2;
feedbackDelays = 1:2;
hiddenLayerSize = 10;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

%Read in 10 seconds of music and global vars
Fs = 44100;
samples = [1,10*Fs];
filePath = "Ludwig_van_Beethoven_-_Concerto_for_Piano_and_Orchestra_no._4_in_G_major_-_3._Rondo_(Wilhelm_Kempff,_Ferdinand_Leitner,_Berliner_Philharmoniker,_1962).flac";
[signal, Fs] = audioread(filePath, samples);

prevSignal = signal(:,1);

%input
X = tonndata(prevSignal,false,false);

%% Data reading
%output object
data = [];
for i = 1:5
    d = daq("ni");
    %Add Channels
    %Add channels and set channel properties, if any.
    chanInput = addinput(d,"Dev1","ai0","Voltage");
    addoutput(d,"Dev1","ao0","Voltage");

    chanInput.Range = [-10,10];
    %Set DataAcquisition Rate
    %Set scan rate.
    d.Rate = Fs;
    amplitudePeakToPeak_ch1 = 2;

    %Generate Signal Data
    %Write the signal data.
    %preload(d, prevSignal);
    %start(d,"repeatoutput");
    %pause(1); % Adjust the duration of signal generation.

    %Read Data
    %Read the data in timetable format.
    data = readwrite(d, prevSignal).Variables;%read(d,seconds(10)).Variables;
    stop(d);

    %Clean Up
    %Clear all DataAcquisition and channel objects.
    clear d;

    % now for nn stuff
    %output
    T = tonndata(data(1:length(prevSignal)),false,false);
    [x,xi,ai,t] = preparets(net,X,{},T);
    [net,tr] = train(net,x,t,xi,ai);
    prevSignal = cell2mat(net(x,xi,ai));
end

%%
filename = 'beethovenData.mat';
%save(filename,'data','outputSignal')
