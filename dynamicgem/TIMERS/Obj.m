function  L = Obj(Sim, U, V)
% Sim is the N x N sparse similarity matrix
% U is the N x K left embedding vector
% V is the N x K right embedding vector
% returns || S - U * V^T ||_F^2

% A trick to reduce space complexity to O(M)

[~,k] = size(U);
PS_u = U' * U;
PS_v = V' * V;
% PS_u PS_v are the K x K matrix, pre-calculated sum for embedding vector
% PS_u(i,j) = sum_k=1^N U(k,i)U(k,j)
% PS_v(i,j) = sum_k=1^N V(k,i)V(k,j)

[temp_row, temp_col, temp_value] = find(Sim);
% calculate first term
L = sum(temp_value .* temp_value);

% calculate second term
M = length(temp_value);
% separated into k iteration to avoid memory overflow
for i = 1:k
    start_index = floor((i - 1) * M / k + 1);
    end_index = floor(i * M / k);
	temp_inner = sum( U(temp_row(start_index:end_index),:) .* V(temp_col(start_index:end_index),:) ,2);
    L = L - 2 * sum(  temp_value(start_index:end_index) .* temp_inner );
end

% calculate third term
L = L + sum(sum(PS_u .* PS_v));
end
