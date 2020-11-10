close all; clc;

l = length(data(:,1));
factor = 10;
t = 10; % seconds
f_s = l/t;
iData = [];
for i = 1:26

    iData = [iData data(l/factor+1:2*l/factor,i)];

end

%%Time specifications:
Fs = 10000;                   % samples per second
dt = 1/Fs;                   % seconds per sample
StopTime = 10;             % seconds
t = (0:dt:StopTime-dt)';     % seconds

%%Sine wave:
Fc = 50; % hertz
oData = [];
for i = 1:26

    sig = sin(2*pi*(Fc+10*(i-1))*t);
    oData = [oData sig(l/factor+1:2*l/factor)];

end 

singleI = iData(:,1);
singleO = oData(:,1);
