close all; clear all; clc;

%length of our noise vector - l corresponds to the number of samples. We are assuming one second (unit time), so this also corresponds to the sampling frequency-
%Minimum sampling frequency, for no aliasing = 2*signal frequency -> Nqwist sampling theorem
duration_of_signal = 1; % in seconds
%sampling rate != frequencies present in signal or noise - completely different frequencies

Fs = 600; %means we're sampling 600 samples/second - our range of frequencies we want to see in the noise is 50Hz->300Hz, everything below 300Hz will be oversampled
%But 600 samples means our sampling frequency is 1/600
NumOfSamples = Fs*duration_of_signal;
period = 1/600;
time = 0:period:duration_of_signal-period;
n = 1:1:100;
%noise matrix
x = zeros(1000, 600);
O = zeros(1000, 600);
Noisy = zeros(1000, 600);
for i = 1:1000
    %We want the duratio of our noise to be 600 samples as well, this
    %doesnt really matter, it just needs to be a bunch of white random
    %numbers and we *say* it all occured within a second
    Y = wgn(1, 600, 0, 1, i); %sin(300*pi*time);%
    x(i,:) = Y;%bandpass(Y(1:600),[50 300],600);
    Noisy(i,:) = Y + 0.01*Y.^3;
    O(i,:) = fft(x(i,:));
end
%%
av = zeros(1,600);

%size(x)
%size(O)

for i = 1:1000
y = O(i,1:end);
% f = (0:length(y)-1)*Fs/length(y);
% 
% plot(f,abs(y))
% title('Magnitude')

n = length(x(1,1:end));                         
fshift = (-n/2:n/2-1)*(Fs/n);
yshift = fftshift(y);

av = av + abs(yshift)./1000;

plot(fshift(301:end),abs(yshift(301:end)))
hold on
end
%%
%%Time specifications:
   Fs = 100;                   % samples per second
   dt = 1/Fs;                   % seconds per sample
   StopTime = 0.25;             % seconds
   t = (0:dt:StopTime-dt);     % seconds
   %%Sine wave:
   Fc = 60;                     % hertz
   x = cos(2*pi*Fc*t);
   % Plot the signal versus time:
   stem(t,x);
   xlabel('time (in seconds)');
   title('Signal versus Time');
   zoom xon;