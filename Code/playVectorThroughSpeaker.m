function output = playVectorThroughSpeaker(vector, Fs)
%vector = vector';
if isa(vector,'dlarray')
    vector = extractdata(vector);
elseif isa(vector, 'gpuArray')
    vector = gather(vector);
end


%Duration of signal, not used as we have a predefined signal length
%s = d;
%Read in 10 seconds of music and global vars
%Fs = 44100;

% we're gonna gonna run a training loop on this and plug the new value into
d = daq("ni");
flush(d)
%Add Channels
%Add channels and set channel properties, if any.
addoutput(d,"Dev1","ao0","Voltage");
chanInput = addinput(d,"Dev1","ai0","Voltage");
chanInput.Range = [-10,10];
%Set DataAcquisition Rate
%Set scan rate.
d.Rate = Fs;

%Generate Signal Data
%Write the signal data.
preload(d, vector);
start(d,"repeatoutput");
%pause(1); % Adjust the duration of signal generation.

%Read Data
%Read the data in timetable format.
output = read(d, length(vector)).Variables;

stop(d);

%Clean Up
%Clear all DataAcquisition and channel objects.
clear sound d;
output = dlarray(output);
end