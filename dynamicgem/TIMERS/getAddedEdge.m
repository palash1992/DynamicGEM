function S_add=getAddedEdge(A,filepath,M)
A_new=parseData(filepath,M);
S_add = sparse(length(A),length(A)); 

[i_new,j_new,val_new]=find(A_new);
for k=1:length(i_new)
 if(full(A(i_new(k),j_new(k)))~=val_new(k))
   S_add(i_new(k),j_new(k))=val_new(k);
 end
end


