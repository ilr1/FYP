function y=nsound(note,duration,muteFlag)
%NSOUND Play a single note on the C Major scale.
% NSOUND plays a single note for a specified duration. NSOUND recognizes
% traditional english notaion, e.g., C, G#, E, etc.
%
% Syntax:  y = nsound(note,duration,[muteFlag])
%
% Arguments:
% note - one of the strings
% {'C3','C#3','D3','D#3','E3','F3','F#3','G3','G#3','A3','A#3','B3',
% {'C4','C#4','D4','D#4','E4','F4','F#4','G4','G#4','A4','A#4','B4',
% {'C5','C#5','D5','D#5','E5','F5','F#5','G5','G#5','A5','A#5','B5'},
% or the character 'p'.
% duration - a numeric scalar.
% muteFlag (optional) - a logical scalar.
%
% Usage and output:
% The default behavior of NSOUND is to create a vector y holding a sine
% wave with a frequency corresponding to note and length corresponding to
% duration, and then send it to MATLAB's SOUND function to produce the note
% on speakers. If 'p' is entered instead of a note symbol a vector of zeros
% is produced and silence is `played' for duration seconds. An optional
% input argument muteFlag can be used to supress the final stage, and just
% return the vector y to the caller.
%
% Examples:
%   nsound('C4',1)
%   plays middle C for one second.
%   nsound('E3',1)
%   plays E below middle C (the bass E string on a guitar).
%   y=nsound('C4',1,true)
%   creates a vector y without playing anything.
%   sound(y)
%   will then sound middle C for one second.
%
% Called m-files: sound.m (MATLAB intrinsic).
%
% Author: Naor
%
% See also sound, nPlay, nTune.

% --- Verify input ---
switch nargin
    case 0
        note='C4';
        duration=1;
        muteFlag=false;
    case 1
        duration=1;
        muteFlag=false;
    case 2
        muteFlag=false;
end

if ~isreal(duration)||~ischar(note)
    error('Wrong type of input argument.')
end
if ~isscalar(duration)||duration<=0
    error('Argument must be positive scalar.')
end

% --- Determine frequency ---
% (American standard pitch. Adopted by the American Standards Association
% in 1936.)
switch note
    case 'C3'
        f=130.81;
    case 'C#3'
        f=138.59;
    case 'D3'
        f=146.83;
    case 'D#3'
        f=155.56;
    case 'E3'
        f=164.81;
    case 'F3'
        f=174.61;
    case 'F#3'
        f=185.00;
    case 'G3'
        f=196.00;
    case 'G#3'
        f=207.65;
    case 'A3'
        f=220.00;
    case 'A#3'
        f=233.08;
    case 'B3'
        f=246.94;
    case 'C4'
        f=261.63;
    case 'C#4'
        f=277.18;
    case 'D4'
        f=293.66;
    case 'D#4'
        f=311.13;
    case 'E4'
        f=329.63;
    case 'F4'
        f=349.23;
    case 'F#4'
        f=369.99;
    case 'G4'
        f=392.00;
    case 'G#4'
        f=415.30;
    case 'A4'
        f=440.00;
    case 'A#4'
        f=466.16;
    case 'B4'
        f=493.88;
    case 'C5'
        f=523.25;
    case 'C#5'
        f=554.37;
    case 'D5'
        f=587.33;
    case 'D#5'
        f=622.25;
    case 'E5'
        f=659.26;
    case 'F5'
        f=698.46;
    case 'F#5'
        f=739.99;
    case 'G5'
        f=783.99;
    case 'G#5'
        f=830.61;
    case 'A5'
        f=880.00;
    case 'A#5'
        f=932.33;
    case 'B5'
        f=987.77;
    case 'p'
        f=000.00;
    otherwise
        error('%s is not a recognized note.',note)
end

% --- Create a pitch vector ---
Fs=8192; % Sampling frequency.
t=0:1/Fs:duration;
y=sin(2*pi*f*t);
if muteFlag, return, end

% --- Sound off ---
sound(y,Fs)