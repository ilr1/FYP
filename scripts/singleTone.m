close all; clear all; clc;   
%%Time specifications:
Fs = 600;                   % samples per second
dt = 1/Fs;                   % seconds per sample
StopTime = 1;             % seconds
t = (0:dt:StopTime-dt)';     % seconds

%%Sine wave:
Fc = 55;                     % hertz
x = cos(2*pi*Fc*t);

% Plot the signal versus time:
figure;
stem(t,x);
xlabel('time (in seconds)');
title('Signal versus Time');
zoom xon;