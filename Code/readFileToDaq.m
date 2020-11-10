function [data, Fs] = readFileToDaq(filePath, initial)

%Duration of signal
s = 20;
%Read in 10 seconds of music and global vars
Fs = 44100;
samples = [1,s*Fs];

% we're gonna gonna run a training loop on this and plug the new value into
if initial == 1
    [signal, Fs] = audioread(filePath, samples);
    prevSignal = signal(:,1);
else
     load(filePath, "data", "Fs");
     prevSignal = data;
end

d = daq("ni");

%Add Channels
%Add channels and set channel properties, if any.
addoutput(d,"Dev1","ao0","Voltage");
chanInput = addinput(d,"Dev1","ai0","Voltage");
chanInput.Range = [-10,10];
%Set DataAcquisition Rate
%Set scan rate.
d.Rate = Fs;

outputSignal = prevSignal;

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
clear sound d;
%write the output of the mic to a file for python to read in
filename = 'pythonData.mat';
save(filename,'data','Fs');
end