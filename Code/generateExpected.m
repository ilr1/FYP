close all; clc;

%Define Output Signal
freqA = 50:1:1000;
thdA = zeros(1,length(freqA));
data = [];

for i = 1:1:length(freqA)
    
   
    d.Rate = 10000;
    
    amplitudePeakToPeak_ch1 = 2;

    sineFrequency = freqA(i); % 100 Hz
    duration = 1;

    outputSignal = [];
    outputSignal(:,1) = createSine(amplitudePeakToPeak_ch1/2, sineFrequency, d.Rate, "bipolar", duration);
    outputSignal = [outputSignal; zeros(1, 10000 - length(outputSignal))'];
    data = [data outputSignal];
end

filename = 'From50HzTo1kHzPureSine.mat';
%save(filename,'')

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