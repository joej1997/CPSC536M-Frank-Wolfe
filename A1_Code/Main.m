%Numerical experiment CPSC536M, HW 2.

%This script is testing Frank-Wolfe Algorithm on an image deblurring
%probelm of the for 0.5*norm(A*X-B)^2_2 such that X is in \tau B_1.
%It uses the function fwdeblur2.m

clear all;
clc;

X=double(imread('mri.tif')); %Original image
X=X/88; %scale so that every entry of the original image is between 0 and 1.
disp(X);
[m,n]=size(X); %Size of the image
[P, center]=psfGaussian([9,9],4); %Apply a Gaussian blur of size 9 x9 with standard deviation 4.
B=imfilter(X,P,'symmetric');  %Apply the PSF P to the original image
Pbig=zeroPSF(P,[m,n]); %Add zeros to P so that the dimension agrees  with the image
[Ar,Ac]=kronfact(Pbig,center,'reflexive');  %Since the probelem is separable, we know A=Ar \otimes Ac

tau=10; %radius of \ell_1-ball
pars.iterMax=20000; % Maximum number of iterations
[Xbest10, funval]=fwdeblur2(B,Ar,Ac,tau,pars);  % run Frank-Wolfe Algorithm

tau=100;
[Xbest100, funval]=fwdeblur2(B,Ar,Ac,tau,pars);  % run Frank-Wolfe Algorithm

tau=1000;
[Xbest1000, funval]=fwdeblur2(B,Ar,Ac,tau,pars);  % run Frank-Wolfe Algorithm
montage([Xbest10 Xbest100 Xbest1000],'size',[1 NaN]);

%The following functions are strongly inspired from the coding available in
%the HNO.zip file available at http://www.imm.dtu.dk/~pcha/HNO/ from
%Hansen.
function Paug = zeroPSF(P, length)
Paug = zeros(length);
[rP, cP]=size(P);
Paug((1:rP),(1:cP))=P;
end

function [PSFmat, center] = psfGaussian(dim, sd)
% Obtain a Gaussian blur point spread function (PSF). 
%  Input:
%        dim:  Dimension  that we want for the PSF matrix.
%         sd:  Vector with standard deviations of the Gaussian in 
%              the vertical and horizontal directions.
%              If s is a scalar, then both standard deviations are s.
%              Default value is 2.
%  Output:
%         PSF:  Matrix containing the point spread function.
%         center:  [row, col] gives index of center of PSF
% Check validity of inputs.
if (nargin < 1)
   error('dim must be specified.')
end
l = length(dim);
if l == 1
  m = dim;
  n = dim;
else
  m = dim(1);
  n = dim(2);
end
if (nargin < 2)
  sd = 2.0; %Default value for s
end
if length(sd) == 1 %If S is a scalar
  sd = [sd,sd];
end
% Set up grid points to compute the Gaussian function.
x = -fix(n/2):ceil(n/2)-1;
y = -fix(m/2):ceil(m/2)-1;
[X,Y] = meshgrid(x,y);
% Compute the Gaussian. Normalize the PSF.
PSFmat = exp( -(X.^2)/(2*sd(1)^2) - (Y.^2)/(2*sd(2)^2) );
PSFmat = PSFmat / sum(PSFmat(:));
% Set up the center
if nargout == 2
  [mm, nn] = find(PSFmat == max(PSFmat(:)));
  center = [mm(1), nn(1)];
end
end

function [Ar, Ac] = kronfact(P, center, BC)
%KRONDECOMP Kronecker product decomposition of a PSF matrix
% Compute the factorization A = kron(Ar, Ac),
% where A is a blurring matrix defined by a PSF array.  The result is
% an approximation if the PSF matrix is not rank-one.
%
%  Input: 
%        P:  Matrix containing the point spread function.
%        center:  [row, col] = indices of center of PSF, P.
%        BC:  String indicating boundary condition.
%             ('zero', 'reflexive', or 'periodic')
%              Default is set to  'zero'.
%
%  Output:
%          Ac, Ar:  Matrices in the Kronecker product decomposition. 
%  Note that the structure of Ac and Ar depends on the BC.
% Check validity of inputs.
if (nargin < 2)
   error('P and center must be input.')
end
if (nargin < 3)
   BC = 'zero'; %If BC not specified
end
% Find the two largest singular values and corresponding singular vectors
% of the PSF. will be used to see if the PSF is separable.
[U, S, V] = svds(P, 2);
if ( S(2,2) / S(1,1) > sqrt(eps) )  
  warning('The PSF is not separable. It will return an approximation.')
end
% Since the PSF has nonnegative entries, we want the the vectors of the
% rank-one decomposition of the PSF to have nonnegative components.  This
% means that the singular vectors corresponding to the largest singular value of P
% should have nonnegative entries.
%  Check if it is true and change sign if necessary.
minU = abs(min(U(:,1)));
maxU = max(abs(U(:,1)));
if minU == maxU
  U = -U;
  V = -V;
end
% The matrices Ar and Ac are defined by vectors r and c. 
% r and c are:
c = sqrt(S(1,1))*U(:,1);
r = sqrt(S(1,1))*V(:,1);
% The structure of Ar and Ac depends on the BC:
switch BC
  case 'zero'
    % Build Toeplitz matrices:
    Ar = consToep(r, center(2));
    Ac = consToep(c, center(1));
  case 'reflexive'
    % Build Toeplitz-plus-Hankel matrices:
    Ar = consToep(r, center(2)) + consHank(r, center(2));
    Ac = consToep(c, center(1)) + consHank(c, center(1));
  case 'periodic'
    % Build circulant matrices:
    Ar = consCirc(r, center(2));
    Ac = consCirc(c, center(1));
  otherwise
    error('Invalid BC.')
end
end
function T = consToep(c, k)
%  Construct a banded Toeplitz matrix from a central column and an index
%  denoting the central column.
n = length(c);
col = zeros(n,1);
row = col';
col(1:n-k+1,1) = c(k:n);
row(1,1:k) = c(k:-1:1)';
T = toeplitz(col, row);
end
function C = consCirc(c, k)
%  Construct a banded circulant matrix from a central column and an index
%  denoting the central column.
n = length(c);
col = [c(k:n); c(1:k-1)];
row = [c(k:-1:1)', c(n:-1:k+1)'];
C = toeplitz(col, row);
end
function H = consHank(c, k)
%  Construct a Hankel matrix for separable PSF and reflexive boundary
%  conditions.
n = length(c);
col = zeros(n,1);
col(1:n-k) = c(k+1:n);
row = zeros(n,1);
row(n-k+2:n) = c(1:k-1);
H = hankel(col, row);
end
