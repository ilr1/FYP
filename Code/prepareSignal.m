close all; clear all; clc;

%load in 5 seconds of traditional audio
Fs = 44100;
fileName = '..\Assets\Ludwig_van_Beethoven_-_Concerto_for_Piano_and_Orchestra_no._4_in_G_major_-_3._Rondo_(Wilhelm_Kempff,_Ferdinand_Leitner,_Berliner_Philharmoniker,_1962).flac';
audio = audioread(fileName,[1 Fs*5]);

[y, d] = lowpass(audio,2500,Fs,'ImpulseResponse','iir','Steepness',0.95);

x = downsample(y,9);
%%
Fs = 8192;
clear chord;
F = ["A4", "C#5", "E5"];
chord = zeros(1, Fs+1);
for i = 1:length(F)
   chord = chord + nsound(char(F(i)),1,1); 
end
chord = chord(1:end-1);
F = ["A4", "C5", "E5"];
chord1 = zeros(1, Fs+1);
for i = 1:length(F)
   chord1 = chord1 + nsound(char(F(i)),1,1); 
end
chord1 = chord1(1:end-1);
chord = chord/max(chord);
chord1 = chord1/max(chord1);

chord = [chord chord1];
