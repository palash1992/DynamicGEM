function [A,E,TimeStamp] = Random_Com(N,M,p,seed,c_num,c_size,c_prob)
% N: number of nodes
% M: numbef of random edges (approximately)
% p: proportion in static matrix, A
% seed: random seed
% c_num: number of communities
% c_size: approximate community size
% c_prob: approximate edge forming probability
% returns random graph + community appearing

% create random network, undirected, unweighted
rng(seed);
temp = sparse(randi(N,round(M/2),1),randi(N,round(M/2),1),1,N,N);
temp = temp - diag(diag(temp));
temp = temp + temp';                 % undirected
[temp_row,temp_col,~] = find(temp);  % get rid of multiple edges 
temp_choose = (temp_row > temp_col);
temp_row = temp_row(temp_choose);
temp_col = temp_col(temp_choose);
clear temp_choose;
% randomly generate order
temp_order = randperm(length(temp_row));
temp_row = temp_row(temp_order);
temp_col = temp_col(temp_order);
clear temp_order;
% split into static and dynamic network
temp_num = round(p * length(temp_row));
A = sparse(temp_row(1:temp_num),temp_col(1:temp_num),1,N,N);
A = A + A';
E = [temp_row(temp_num + 1:end),temp_col(temp_num + 1:end)];
% randomly generate a timestamp
TimeStamp = sort(rand(length(E),1),'ascend');

% create community edge
% temp: store all existings edges to ensure no overlapping
c_add = [];
for i = 1:c_num
    c_node = randperm(N,c_size);
    c_temp = sparse(N,N);
    c_temp(c_node,c_node) = 1;
    c_temp = c_temp - c_temp .* (temp > 0);
    [temp_row,temp_col] = find(c_temp);
    temp_choose = (temp_row < temp_col);
    temp_row = temp_row(temp_choose);
    temp_col = temp_col(temp_choose);
    temp_choose = (rand(length(temp_row),1) <= c_prob);
    temp_row = temp_row(temp_choose);
    temp_col = temp_col(temp_choose);
    temp_order = randperm(length(temp_row));
    temp_row = temp_row(temp_order,:);
    temp_col = temp_col(temp_order,:);
    c_add = [c_add;temp_row,temp_col];
    temp(sub2ind([N,N],temp_row,temp_col)) = 1;
    temp(sub2ind([N,N],temp_col,temp_row)) = 1;
end
temp_insert = round(rand(1,1) * length(E) * 0.6); % avoid too late change
% create a simulated time
t_add = sort(rand(length(c_add),1),'ascend');
t_add = TimeStamp(temp_insert) + t_add * (TimeStamp(temp_insert + 1) - TimeStamp(temp_insert));
E = [E(1:temp_insert,:);c_add;E(temp_insert+1:end,:)];
TimeStamp = [TimeStamp(1:temp_insert);t_add;TimeStamp(temp_insert+1:end)];

disp(['Node number:' num2str(N) '; Edge number ' num2str(sum(sum(A > 0))) '; New edge number:' num2str(2*length(E)) '(Community:' num2str(2*length(c_add)) ')']);
end