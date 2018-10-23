function S_delta=deltaA_sbm(A,filepath,M)
A_new=parseData_sbm(filepath,M);
S_delta = sparse(length(A),length(A)); 

[i_old,j_old,val_old]=find(A);
[i_new,j_new,val_new]=find(A_new);

for k=1:length(i_new)
 temp_old=full(A(i_new(k),j_new(k)));
 if(temp_old<val_new(k))
   S_delta(i_new(k),j_new(k))=val_new(k)-temp_old;
 elseif (temp_old>val_new(k))
   S_delta(i_new(k),j_new(k))=val_new(k)-temp_old; % need to make sure if sign matters!
 else

 end
end

for k=1:length(i_old)
 temp_new=full(A_new(i_old(k),j_old(k)));
 if(temp_new<val_old(k))
   S_delta(i_old(k),j_old(k))=temp_new-val_old(k);
 elseif (temp_new>val_old(k))
   S_delta(i_old(k),j_old(k))=temp_new-val_old(k); % need to make sure if sign matters!
 else

 end
end

end

