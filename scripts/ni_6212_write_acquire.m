%daq.getDevices   %check for devices
s=daq.createSession('ni'); %Assume an ni device
ch=addAnalogInputChannel(s,'Dev1',0,'Voltage');  % add Ch0
%s.DurationInSeconds = 4;% length of acquisition
ch.Range=[-10,10];
% % 
 s.Rate=40000;  % rate
 rate=s.Rate;

 
 % analogue output
 ao=addAnalogOutputChannel(s,'Dev1',0,'Voltage');  % add Ch0


% t=linspace(0,1,rate)';  
% y=cos(2*pi*t*3);
%outputdata=[out;0];  % set output to zero after writing
queueOutputData(s,outputdata);
data=startForeground(s);



%  disp('Started');
% [datad,timestamps]=startForeground(s);