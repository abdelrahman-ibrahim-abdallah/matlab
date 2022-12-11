# matlab
z transform
Example 1:

syms z n
a=ztrans(1/16^n)
 

Example 2:

syms Z n
b=iztrans(3*Z/(Z+1))

 
Example 3:

 

b=[0 1 1 ]
a= [1 -2 +3]
roots(a)
roots(b)
zplane(b,a);

Example 4:
clc ;
close all;
syms n;
a=2;
x=a^n;
X=ztrans(x); %finding z transform
disp('z tranform of a^n a>1');
disp(X);
syms n;
a=0.5;
x=a^n;
X1=ztrans(x);
disp('z tranform of a^n 0<a<1');
disp(X1);
syms n;
a=2;
x=1+n;
X2=ztrans(x);
disp('z tranform of 1+n');
disp(X2);
A=iztrans(X);
disp('inverse z tranform of a^n a>1');
disp(A);
B=iztrans(X1);
disp('inverse z tranform of a^n 0<a<1');
disp(B);
C=iztrans(X2);
disp('inverse z tranform of 1+n');
disp(C);
subplot(1,3,1);
zplane([1 0],[1 -2]);
subplot(1,3,2);
zplane([1 0],[1 -1/2]);
subplot(1,3,3);
zplane([1  0 0],[1 -2 1]);

Example 5:

Express the following z-transform in factored form , plot its poles and zeros, and then determine its ROCs.
             2z4+16z3+44z2+56z+32
G(z)=  -------------------------------- 
        3z4+3z3-15z2+18z-12


num = input('Type in the numerator coefficients =');
 den = input('Type in the denominator coefficients =');
 
% num=[2 16 44 56 32]; den=[3 3 -15 18 -12];
 [z,p,k]=tf2zp(num,den);
 m = abs(p) % to find distance from origin of poles
 disp('Zeros are at'); disp(z);
 disp('poles are at'); disp(p);
 disp('gain constant'); disp(k);
 disp('Radius of poles'); disp(m);
 sos= zp2sos(z,p,k);
 disp('Second order sections');disp(real(sos));
 zplane(num,den) 
---------------------------------------------------------------------------------------------------------

Loading Images, converting color image to grayscale, writing the image into a file
A='D:\.. ..\img1.bmp';
B=imread(A,'bmp');
figure(1),imshow(B);
C=rgb2gray(B);
figure(2),imshow(C);
imwrite(C,'C:\Users\pc\Desktop\.. ..\imgwrite1.jpg','jpg');

A = im2double(B)
%{
I2 = im2double(I) converts the intensity image I to double precision, 
rescaling the data if necessary. 
I can be a grayscale intensity image, a truecolor image, or a binary image.
If the input image is of class double, 
then the output image is identical.
%}
H=im2double(imread(A,'bmp'));

%{
Adjust image intensity values or colormap
 = imadjust(I,[low_in; high_in],[low_out; high_out]) maps the values in I to new values in J 
such that values between low_in and high_in map to values between low_out and high_out.
Note   If high_out is less than low_out, 
imadjust reverses the output image, as in a photographic negative.
%}
C=imadjust(B,[0 1],[1 0]);

%Enhance contrast using histogram equalization
E=histeq(B);

%{
level = graythresh(I) computes a global threshold (level)  that can be used to convert an intensity image to a binary image 
with im2bw. 
level is a normalized intensity value that lies in the range [0, 1].
%}
F = graythresh(B);

%{
BW = im2bw(I, level) 
converts the grayscale image I to a binary image. 
The output image BW replaces all pixels in 
the input image with luminance greater than level with the value 1 (white) 
and replaces all other pixels with the value 0 (black)
%}
G= im2bw(B,F);

A='D:\.. .. \img1.bmp';
H=im2double(imread(A,'bmp'));
figure(1), imshow(H);
B= rgb2gray(H);
figure(2), imshow(B);
C=imadjust(B,[0 1],[1 0]);
figure(3),imshow(C);
E=histeq(B);
figure(4), imshow(E);
F = graythresh(B);
G= im2bw(B,F);
figure(5), imshow(G);

Adding salt and pepper noise to an image
II = imread('eight.tif'); I=rgb2gray(II);
J = imnoise(I,'salt & pepper');
figure, imshow(I)
figure, imshow(J)

Filtering noise using median filter
II = imread('eight.tif');I=rgb2gray(II);
J = imnoise(I,'salt & pepper');
figure, imshow(I)
figure, imshow(J)
k=medfilt2(J);figure, imshow(k);

I=imread('rice.tif'); % load MATLAB image
subplot(2,2,1);
imshow(I); title('original image');
h=fspecial('sobel');
A=imfilter(I,h);
subplot(2,2,2);
imshow(A);title('horizontal sobel');
B=edge(I,'sobel',[],'both');
subplot(2,2,3);imshow(B);title('sobel');
C=edge(I,'canny',[],1);
subplot(2,2,4);imshow(C);title('canny');


}ii=imread('eight.tif'); 
i=rgb2gray(ii);
h=fspecial('sobel'); % creat sobel hz edge detection
a=imfilter(i,h);% to get vertival direction use h’
b=edge(i,'sobel',[],'both');% filter sobel in both directions
c=edge(i,'canny',[],1);% canny edge detection and 1 is the standard deviation of gaussian filter
subplot(2,2,1),imshow(i);% original image
subplot(2,2,2),imshow(a);% hz sobel
subplot(2,2,3), imshow(b);% sobel both directions
subplot(2,2,4), imshow(c);% canny edge detection
--------------------------------------------------------------------------------------------
signal speech

#y = sin(2*pi*50*t) + 2*sin(2*pi*120*t);
t = (0:0.001:1)';
y = sin(2*pi*50*t) + 2*sin(2*pi*120*t);
yn = y + 0.5*randn(size(t));% adding white noise
figure
subplot(2,1,1); 
plot(t(1:50),y(1:50),'r')
subplot(2,1,2);
plot(t(1:50),yn(1:50),'b')
#Draw a stem plot
t = 0:0.01:2; % sample points from 0 to 2 in steps of 0.01
xt = sin(2*pi*t); % Evaluate sin(2 pi t)
n = 0:1:40; % sample index from 0 to 40
xn = sin(0.1*pi*n); % Evaluate sin(0.2 pi n)
Hs = stem(n,xn,'b', 'filled'); % Stem-plot with handle Hs
set(Hs,'markersize',4); % Change circle size
xlabel('n'); ylabel('x(n)'); % Label axis
title('Stem Plot of sin(0.2 pi n)'); % Title plot
figure
plot(t,xt,'b'); hold on; % Create plot with blue line
Hs = stem(n*0.05,xn,'b','filled'); % Stem-plot with handle Hs
set(Hs,'markersize',4); hold off; % Change circle size
#t = 0:0.01:2; % sample points from 0 to 2 in steps of 0.01
xt = sin(2*pi*t); % Evaluate sin(2 pi t)
n = 0:1:40; % sample index from 0 to 20
xn = sin(0.1*pi*n); % Evaluate sin(0.2 pi n)
figure
subplot(2,1,1); % Two rows, one column, first plot
plot(t,xt,'b'); % Create plot with blue line
 
subplot(2,1,2); % Two rows, one column, second plot
Hs = stem(n,xn,'b','filled'); % Stem-plot with handle Hs

Impulse, Step, and Ramp, Quad, square Functions
t = (-1:0.01:1)';
impulse = t==0;
unitstep = t>=0;
ramp = t.*unitstep;
quad = t.^2.*unitstep;
sqwave = 0.81*square(4*pi*t);
figure
subplot(5,1,1); plot(t,impulse,'r'); % impule
subplot(5,1,2); plot(t,unitstep,'b'); % unit step
subplot(5,1,3); plot(t,ramp,'k'); % ramp
subplot(5,1,4); plot(t,quad,'g'); % quad
subplot(5,1,5); plot(t,sqwave,'b')% square
Exponential signals
n3 = 0:0.5:10; x3 = (0.9).^n3;
figure
Hs = stem(n3,x3,'b','filled');
complex valued exponential
n4 = 5:0.2:10; x4 = exp((1+8j)*n4);
figure
Hs = stem(n4,x4,'b','filled');

sinusoidal sequence
n5 = 0:0.2:10; x5 = 3*cos(0.1*pi*n5+pi/3) + 2*sin(0.5*pi*n5);
figure
Hs = stem(n5,x5,'b','filled');

Sinc Function
x = linspace(-5,5);
y = sinc(x);
figure
plot(x,y)
grid

signal addition
n1 = 0:20; x1 = (0.9).^n1;
n2 = 5:30; x2 =3*cos(0.1*pi*n2+pi/3)+ 2*sin(0.5*pi*n2);
nn = min(min(n1),min(n2)):max(max(n1),max(n2)); % duration of y(n)
y1 = zeros(1,length(nn)); y2 = y1;
% x1 with duration of y
y1(find((nn>=min(n1))&(nn<=max(n1))==1))=x1;  
% x2 with duration of y
y2(find((nn>=min(n2))&(nn<=max(n2))==1))=x2; 
y = y1+y2; % sequence addition
figure
subplot(3,1,1); Hs = stem(nn,y1,'r','filled'); 
subplot(3,1,2); Hs = stem(nn,y2,'g','filled'); 
subplot(3,1,3); Hs = stem(nn,y,'b','filled');
figure
hold on
Hs = stem(nn,y1,'r','filled'); 
Hs = stem(nn,y2,'g','filled'); 
Hs = stem(nn,y,'b','filled');
hold off

signal multiplication
n1 = 0:20; x1 = (0.9).^n1;
n2 = 5:30; x2 =3*cos(0.1*pi*n2+pi/3)+ 2*sin(0.5*pi*n2);
nn = min(min(n1),min(n2)):max(max(n1),max(n2)); % duration of y(n)
y1 = zeros(1,length(nn)); y2 = y1;
% x1 with duration of y
y1(find((nn>=min(n1))&(nn<=max(n1))==1))=x1;  
% x2 with duration of y
y2(find((nn>=min(n2))&(nn<=max(n2))==1))=x2; 
y = y1.*y2; % sequence multiplication

signal shifting
n4 = 5:30; 
x4=3*cos(0.1*pi*n4+pi/3)+ 2*sin(0.5*pi*n4);
k=5;             m=n4+k;% shift
x4_new =3*cos(0.1*pi*m+pi/3)+ 2*sin(0.5*pi*m);
figure
subplot(2,1,1); Hs = stem(n4,x4,'r','filled'); 
subplot(2,1,2); Hs = stem(m,x4_new,'g','filled'); 
figure
hold on
Hs = stem(n4,x4,'b','filled'); 
Hs = stem(m,x4_new,'g','filled'); hold off

signal folding
n = 5:0.2:20; 
x=3*cos(0.1*pi*n+pi/3)+ 2*sin(0.5*pi*n);
y = fliplr(x); 
figure
subplot(2,1,1); 
Hs = stem(n,x,'r','filled'); 
subplot(2,1,2); 
Hs = stem(n,y,'b','filled'); 

signal energy
n = 5:0.2:20; 
x=3*cos(0.1*pi*n+pi/3)+ 2*sin(0.5*pi*n);
% the value of the signal energy
Ex = sum(abs(x) .^ 2);

Signal convolution
x = [3, 11, 7, 0, -1, 4, 2]; nx = [-3:3];
 h = [2, 3, 0, -5, 2, 1]; nh = [-1:4];
 nyb = nx(1)+nh(1);
nye = nx(length(x)) + nh(length(h));
ny = [nyb:nye];
y = conv(x,h);% without above 3 lines it will make convolution but can't draw result
figure
subplot(2,2,1); stem( nx,x); title('first sequence')
xlabel('n'); ylabel('x(n)'); 
subplot(2,2,2); stem(nh,h); title('second sequence')
xlabel('n'); ylabel('h(n)'); 
subplot(2,2,4); stem(ny,y); title('convolution result')
xlabel('n'); ylabel('y(n)'); 

Periodogram power spectral density estimate
Fs = 1000;     t = 0:1/Fs:1;
x = cos(2*pi*100*t)+randn(size(t));
[Pxx,F] = periodogram(x,[],length(x),Fs);
figure
subplot(3,1,1); plot(t,x);
xlabel('Time');ylabel('x(t)');
title('Original signal')
subplot(3,1,2); plot(F,Pxx)
xlabel('Frequency');
ylabel('Power Spectrum Magnitude');
title('Power spectral Density')
subplot(3,1,3); plot(F,10*log10(Pxx))
xlabel('Frequency');
ylabel(' 'Power Spectrum Magnitude(dB)');
title('Power spectral Density')

Recording an audio signal
Fs=8000;
recObj = audiorecorder
disp('Start speaking.')
recordblocking(recObj, 5);
disp('End of Recording.');
play(recObj);
x = getaudiodata(recObj);
[Pxx,F] = periodogram(x,[],length(x),Fs);
figure
subplot(3,1,1); plot(x);xlabel('Time');ylabel('x(t)');
title('Original sound signal')
subplot(3,1,2); plot(F,Pxx);xlabel('Frequency'); ylabel('PSM');title('Power spectral Density')
subplot(3,1,3); plot(F,10*log10(Pxx))
xlabel('Frequency');ylabel('PSM (dB)');
title('Power spectral Density')

m2
Fs=8000;
recObj = audiorecorder
disp('Start speaking.')
recordblocking(recObj, 5);
disp('End of Recording.');
play(recObj);   %  or sound
x = getaudiodata(recObj);
filename = 's1.wav';
audiowrite(filename,x,Fs);
[y,Fs] = audioread(filename);

soundsc(y,Fs);% Scale data and play as sound

•Use the audioread function to read the file, handel.wav. The audioread function can support WAVE, OGG, FLAC, AU, MP3, and MPEG-4 AAC files.
[y,Fs] = audioread('handel.wav');
•Play the audio.
sound(y,Fs)

Discriminating two sounds based on energy
Fs=8000;
recObj = audiorecorder;
disp('Start speaking1.')
recordblocking(recObj, 3);
disp('End of Recording1.');
x = getaudiodata(recObj);
filename = 's1.wav';   audiowrite(filename,x,Fs);
[y1,Fs] = audioread(filename);
Ey1 = sum(abs(y1) .^ 2);
disp('Start speaking2.') % second signal
recordblocking(recObj, 3);
disp('End of Recording2.');
x2 = getaudiodata(recObj);
filename = 's2.wav';    audiowrite(filename,x2,Fs);
[y2,Fs] = audioread(filename);
Ey2 = sum(abs(y2) .^ 2);
soundsc(y2,Fs);

Filter design (lowpass)
[x,fs] = wavread('s1.wav'); % load audio file
[Pxx1,F1] = periodogram(x,[],length(x),Fs);
% design window based FIR filter stages=200,CF=300
b = fir1(200,300/(fs/2)); 
y = filter(b,1,x); % filter the signal
[Pxx2,F2] = periodogram(y,[],length(y),Fs);
figure
subplot(4,1,1);plot(x)
subplot(4,1,2); plot(F1,Pxx1)
subplot(4,1,3);plot(y)
subplot(4,1,4); plot(F2,Pxx2)

Filter design (hightpass)
[x,fs] = wavread('s1.wav'); % load audio file
[Pxx1,F1] = periodogram(x,[],length(x),Fs);
% design window based FIR filter stages=200,CF=300
b = fir1(200,300/(fs/2),'high');
[Pxx2,F2] = periodogram(y,[],length(y),Fs);
figure
subplot(4,1,1);plot(x)
subplot(4,1,2); plot(F1,Pxx1)
subplot(4,1,3);plot(y)
subplot(4,1,4); plot(F2,Pxx2)

Comparing lowpass and highpass
FS=1000;t = 0:1/Fs:1;
x=cos(2*pi*100*t)+cos(2*pi*500*t)+randn(size(t));
[Pxx,F] = periodogram(x,[],length(x),Fs);
b1 = fir1(200,300/(FS/2)); 
y1 = filter(b1,1,x); % lowpass filter
[Pxx1,F1] = periodogram(y1,[],length(y1),Fs);
b2 = fir1(200,300/(FS/2),'high');
y2 = filter(b2,1,x); % highpass filter 
[Pxx2,F2] = periodogram(y2,[],length(y2),Fs);
figure
subplot(3,1,1);plot(t,x); 
xlabel('time');ylabel('x(t)');title('original signal in time domain');
subplot(3,1,2);plot(t,y1);
xlabel('time');ylabel('y1(t)');title('signal in time domain after removing high freq. complnents');
subplot(3,1,3);plot(t,y2)
xlabel('time');ylabel('y2(t)');title('signal in time domain after removing low freq. complnents');
figure
subplot(3,1,1);plot(F,Pxx); 
xlabel('frequency');ylabel('PSM');title('Power Spectral Density of original signal');
subplot(3,1,2); plot(F1,Pxx1);
xlabel('frequency');ylabel('PSM');title('Power Spectral Density of signal after lowpass filter');
subplot(3,1,3); plot(F2,Pxx2);
xlabel('frequency');ylabel('PSM');title('Power Spectral Density of original signal after highpass filter');
