close all; clear all; clc;
amp=1; 
fs=20500;  % sampling frequency
duration=5;
freq=440;
values=0:1/fs:duration;
a=amp*sin(2*pi* freq*values);
sound(a)