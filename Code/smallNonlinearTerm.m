fs = 20000;
x = linspace(0,1,fs);

y1 = 2*cos(2*pi*100*x);

y2 = 0.5*cos(2*pi*203*x);

y3 = 0.3*cos(2*pi*313*x);

y = y1 + y2 + y3;
y = y + 0.1*y.^2;

ft = fft(y);

subplot(2,1,1)
plot(x,y)
subplot(2,1,2)
f = (0:length(ft)-1)*fs/length(ft); %same as 0:fs/length(f)(N):fs*N-fs/length(f)(N)
plot(f,abs(ft));


n = length(x);                         
fshift = (-n/2:n/2-1)*(fs/n);
yshift = fftshift(ft);
subplot(2,1,2)
plot(fshift,abs(yshift))