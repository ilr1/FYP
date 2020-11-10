%Auto-generated by Data Acquisition Toolbox Analog Output Generator on 22-Jul-2020 11:29:20
%Create DataAcquisition Object
%Create a DataAcquisition object for the specified vendor.
d = daq("ni");

%Add Channels
%Add channels and set channel properties, if any.
addoutput(d,"Dev1","ao0","Voltage");

%Set DataAcquisition Rate
%Set scan rate.
d.Rate = 10000;

%Define Output Signal
%Apply the specified scale and offset on the selected variable to define the output signal. Also, repeat the output signal to ensure that there is at least 0.5 seconds of data.
outputSignal = [];
outputSignal(:,1) = DAQ_3.Dev1_ai0 * 1 + 0;
numCycles = ceil((d.Rate/2)/length(outputSignal));
outputSignal = repmat(outputSignal, numCycles, 1);

%Generate Signal Data
%Write the signal data.
preload(d,outputSignal);
start(d,"repeatoutput")
pause(1) % Adjust the duration of signal generation.
stop(d)

%Clean Up
%Clear all DataAcquisition and channel objects.
clear d outputSignal
