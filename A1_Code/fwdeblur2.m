%This function implements fwid (Frank-Wolfe Algorithm) for solving the
%image deblurring problem 
%It is based on the paper by Beck and Taboulle: A fast iterative
%shrinkage-threshold Algorithm  for linear inverse problems, 2009.

%Input:
%      Iobs: the observed image (blurred and noisy)
%      Ar,Ac: Kronecker Decomposition of the blur matrix A
%      center: A vector of length 2 containing the center of the PSF
%       tau:  radius of the l1-norm ball
%       param:
%              param.iterMax:   maximum number of iterations (default:100)
%Output: Xbest: solution of the problem min ||A(X)-Iobs||^2 s.t X in
%               \tau B_1 (B_1 is l1-norm unit ball)
%        funval_all: Vector containing all function values obtained at each
%                    iteration
function [Xbest,funcval_all]=fwdeblur2(Iobs,Ar,Ac,tau,pars)
%% Assigning parameters based  on param. and default values
flag=exist('pars');
if (flag && isfield(pars,'iterMax'))
    iterMax=pars.iterMax;
else
    iterMax=100;
end
% If there are two outputs, initalize the function value vec.
if (nargout==2)
    funcval_all=[];
end
[m,n]=size(Iobs);
%% initialization
X_iter=zeros(m,n);
fun_val0=0.5*norm(Ac*X_iter*Ar'-Iobs)^2; %Function value at the zero matrix
fprintf('------------------------\n');
fprintf('******FRANK-WOLFE*******\n');
fprintf('------------------------\n');
fprintf('iteration  function value\n');
for i=0:iterMax
    if i==0
        func_val=fun_val0;
        fprintf('%3d    %15.5f \n',i,func_val);
        continue;
    end
     Ej=zeros(m,n); %initialize the the matrix Ej
    % Gradient Matrix: 
     gradXiter=Ac'*(Ac*X_iter*Ar'-Iobs)*Ar;
    [val,index]=max(abs(gradXiter)); %Find the max of each column 
    % Solve the linear subproblem of Frank-Wolfe:
    for j=1:n
       Sign=sign(gradXiter(index(j),j)); %Store the sign
       if Sign==1 
           Ej(index(j),j)=-1; %Take the opposite of the sign
       elseif Sign==-1
           Ej(index(j),j)=1; 
       else
            Ej(index(j),j)=0;
       end
    end
    S=tau*Ej; %Solution of the  linear subproblem
    gamma=2/(i+2); %Step size
    X_iter=(1-gamma)*X_iter+gamma*S; %Update X_iter
    func_val=0.5*norm(Ac*X_iter*Ar'-Iobs)^2; %Compute function value at X_iter
    if (nargout==2)
        funcval_all=[funcval_all;func_val]; %Update vector containing function values
    end
    fprintf('%3d    %15.5f \n',i,func_val); % printing the information of the current iteration
end
Xbest=X_iter; %Set Xbest as the X_iter obtained at iterMax.
end