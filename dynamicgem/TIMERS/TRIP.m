function [New_U,New_S,New_V] = TRIP(Old_U,Old_S,Old_V, Delta_A)
% update using TRIP method
% reference: Chen Chen, and Hanghang Tong. "Fast eigen-functions tracking on dynamic graphs." SDM, 2015.
[N,K] = size(Old_U);

% solve eigenvalue and eigenvectors from SVD, denote as L, X
Old_X = Old_U;    
for i = 1:K    % unify the sign
    [~,temp_i] = max(abs(Old_X(:,i)));
    if (Old_X(temp_i,i) < 0)
        Old_X(:,i) = - Old_X(:,i);
    end
end
[temp_v,temp_i] = max(Old_U);
temp_sign = sign(temp_v .* Old_V(sub2ind([N,K],temp_i,1:K)));  % use maximum absolute value to determine sign
Old_L = diag(Old_S)' .* temp_sign;  % 1 x k eigenvalues
clear temp_v temp_i temp_sign;

% calculate sum term
temp_sum = Old_X' * Delta_A * Old_X;
% calculate eigenvalues change
Delta_L = diag(temp_sum)';
% calculate eigenvectors change
Delta_X = zeros(N,K);
for i = 1:K
   temp_D = diag(ones(1,K) * (Old_L(i) + Delta_L(i)) - Old_L);
   temp_alpha = pinv(temp_D - temp_sum) * temp_sum(:,i);
   Delta_X(:,i) = Old_X * temp_alpha;
end

% return updated result
New_U = (Old_X + Delta_X);
for i = 1:K
    New_U(:,i) = New_U(:,i) ./ sqrt(New_U(:,i)' * New_U(:,i));
end
New_S = diag(abs(Old_L + Delta_L));
New_V = New_U * diag(sign(Old_L + Delta_L));
end
