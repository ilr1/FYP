function y = passSignalThroughTF(x, t)
%%create a transfer function model of a speaker with defined parameters
num = [3 1];
den = [1 0 -16];
sys = tf(num,den);

fs = 1e4;
f = 10;
%t = 0:1/fs:1;
%x = cos(2*pi*f*t);

y = lsim(sys,x,t);
%y = y/max(y); %normalised
%y = y + 5*y.^3;
%plot(t,x,t,y);
%ylim([min(min(x),min(y)) max(max(x),max(y))]);
%legend('input','output');
%set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.5, 0.5, 0.5]);
%set(0,'DefaultFigureWindowStyle','docked');

end