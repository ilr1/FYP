close all; clear all; clc;   
%%Time specifications:
Fs = 600;                   % samples per second
dt = 1/Fs;                   % seconds per sample
StopTime = 1;             % seconds
t = (0:dt:StopTime-dt)';     % seconds

%%Sine wave A1:
F1 = 55;                     % hertz
x1 = cos(2*pi*F1*t);

%%Sine wave E1:
F2 = 82.41;                     % hertz
x2 = cos(2*pi*F2*t);

x = x1+x2;
% Plot the signal versus time:
figure;
plot(t,x);
xlabel('time (in seconds)');
title('Signal versus Time');
zoom xon;