
s = createSine(1, 50, 10000, "bipolar", duration);
plot(s);
function sine = createSine(A, f, sampleRate, type, duration)

numSamplesPerCycle = floor(sampleRate/f);
T = 1/f;
timestep = T/numSamplesPerCycle;
t = (0 : timestep : T-timestep)';

if type == "bipolar"
    y = A*sin(2*pi*f*t);
elseif type == "unipolar"
    y = A*sin(2*pi*f*t) + A;
end

numCycles = round(f*duration);
sine = repmat(y,numCycles,1);
end