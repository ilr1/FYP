close all; clc;

%Define Output Signal
freqA = 50:50:3000;
thdA = zeros(1,length(freqA));
data = [];
for i = 1:1:length(freqA)
    %Auto-generated by Data Acquisition Toolbox Analog Output Generator on 22-Jul-2020 11:29:20
    %Create DataAcquisition Object
    %Create a DataAcquisition object for the specified vendor.
    d = daq("ni");


    %Add Channels
    %Add channels and set channel properties, if any.
    addoutput(d,"Dev1","ao0","Voltage");

    %Add Channels
    %Add channels and set channel properties, if any.
    chanInput = addinput(d,"Dev1","ai0","Voltage");
    chanInput.Range = [-10,10];
 
    %Set DataAcquisition Rate
    %Set scan rate.
    d.Rate = 10000;
    
    %Apply the specified scale and offset on the selected variable to define the output signal. Also, repeat the output signal to ensure that there is at least 0.5 seconds of data.
    %outputSignal = [];
    %outputSignal(:,1) = DAQ_3.Dev1_ai0 * 1 + 0;
    %numCycles = ceil((d.Rate/2)/length(outputSignal));
    %outputSignal = repmat(outputSignal, numCycles, 1);

    amplitudePeakToPeak_ch1 = 2;

    sineFrequency = freqA(i); % 100 Hz
    duration = 1;

    outputSignal = [];
    outputSignal(:,1) = createSine(amplitudePeakToPeak_ch1/2, sineFrequency, d.Rate, "bipolar", duration);


    %Generate Signal Data
    %Write the signal data.
    preload(d,outputSignal);
    start(d,"repeatoutput");
    pause(1); % Adjust the duration of signal generation.

    %Read Data
    %Read the data in timetable format.
    data = [data read(d,seconds(10)).Variables];

    stop(d);

    %Clean Up
    %Clear all DataAcquisition and channel objects.
    clear d outputSignal;

    %sample = DAQ_4.Variables;plot(abs(fft(sample)));
    %hold on;
    %thdA(i) = thd(DAQ_4.Variables,10^4,60,'power');
end
%Plot Data
%Plot the read data on labeled axes.
%plot(DAQ_4.Time, DAQ_4.Variables)
%xlabel("Time")
%ylabel("Amplitude (V)")
%legend(DAQ_4.Properties.VariableNames)







function sine = createSine(A, f, sampleRate, type, duration)

numSamplesPerCycle = floor(sampleRate/f);
T = 1/f;
timestep = T/numSamplesPerCycle;
t = (0 : timestep : T-timestep)';

if type == "bipolar"
    y = A*sin(2*pi*f*t);
elseif type == "unipolar"
    y = A*sin(2*pi*f*t) + A;
end

numCycles = round(f*duration);
sine = repmat(y,numCycles,1);
end