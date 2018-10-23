function Loss_Bound = RefineBound(S_ori, S_add, Loss_ori, K)
% S_ori is n x n symmetric sparse original similarity matrix
% S_add is n x n symmetric sparse new similarity, may overlap with S_ori
% Loss_ori is the value or lower bound of loss for S_ori
% K is embedding dimension
% Return a lower bound of loss for S_ori + S_add by matrix perturbation inequality

% In short, Loss_Bound = Loss_ori + trace_change(S * S^T) - eigs(delta(S *S^T),K)
% Check our paper for detail: 
% Zhang, Ziwei, et al. "TIMERS: Error-Bounded SVD Restart on Dynamic Networks". AAAI, 2018.

% Calculate trace change
S_overlap = (S_add ~= 0) .* S_ori ;
S_temp = S_add + S_overlap;
trace_change = sum(sum(S_temp .* S_temp)) - sum(sum(S_overlap .* S_overlap));
clear S_overlap S_temp; 

% Calculate eigenvalues sum of delta(S * S^T)
  % Notice we only need to deal with non-zero rows/columns
S_temp = S_ori * S_add;
S_temp = S_temp + S_temp' + S_add * S_add;
[~,S_choose,~] = find(sum(S_temp));
S_temp = S_temp(S_choose,S_choose);
clear S_choose; 
% note eigs return largest absolute value, instead of largest 
temp_eigs = eigs(S_temp,min(round(2 * K),length(S_temp))); 
temp_eigs = temp_eigs(temp_eigs >= 0);
temp_eigs = sort(temp_eigs,'descend');
if (length(temp_eigs) >= K)
    eigen_sum = sum(temp_eigs(1:K));
else        % if doesn't calculate enough, add another inequality 
    temp_l = length(temp_eigs);
    eigen_sum = sum(temp_eigs) + temp_eigs(temp_l) * (K - temp_l);
end

% Calculate loss lower bound
Loss_Bound = Loss_ori + trace_change - eigen_sum;
end