function loss_new = Obj_SimChange(S_ori, S_add, U, V, loss_ori)
% S_ori is n x n symmetric sparse original similarity matrix
    % Note, S_ori is last similarity matrix, not static one
% S_add is n x n symmetric sparse new similarity, may overlap with S_ori
% U/V: n x k left/right embedding vectors
% K: embeddiing diemsnion
% loss_ori: original loss, i.e. ||S_ori - U * V^T||_F^2
% return new loss, i.e ||S_ori + S_add - U * V^T||_F^2

[N,K] = size(U);
[temp_row, temp_col, temp_value] = find(S_add);
M = length(temp_value);
temp_old_value = S_ori(sub2ind([N N],temp_row,temp_col));
loss_new = loss_ori;

% avoid memory overflow
for i = 1:K
    start_index = floor((i - 1) * M / K + 1);
    end_index = floor(i * M / K);
    temp_inner = sum( U(temp_row(start_index:end_index),:) .* V(temp_col(start_index:end_index),:),2);
    loss_new = loss_new - sum( (temp_old_value(start_index:end_index) - temp_inner).^2);
    loss_new = loss_new + sum( (temp_old_value(start_index:end_index) + temp_value(start_index:end_index) - temp_inner).^2);
end

end