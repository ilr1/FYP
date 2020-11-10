function y = passSignalThrough_NL_TF(x, t)
%%create a transfer function model of a speaker with defined parameters
num = [1 1 1];
den = [1 1 1];
sys = tf(num,den);

y = lsim(sys,x,t);

y = y + 0.5*y.^3;

%y = y + 0.50*y.^3;


%plot(t,x,t,y);
%ylim([min(min(x),min(y)) max(max(x),max(y))]);
%legend('input','output');
%set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.5, 0.5, 0.5]);
%set(0,'DefaultFigureWindowStyle','docked');

end


% %%
% t = 0:1/fs:10;
% x = cos(20*2*pi*t) + 0.5*(cos(40*2*pi*t));
% plot(abs(fft(passSignalThroughTF(x,t))))
% t = 0:1/fs:1-1/fs;
%t = 0:1/fs:10-1/fs;
%plot(abs(fft(passSignalThroughTF(x,t))))
%x = cos(20*2*pi*t) + 0.5*(cos(40*2*pi*t));
%plot(abs(fft(passSignalThroughTF(x,t))))