function A=splitData_sbm(A,fileLine,M)
tempNodes=strsplit(fileLine,' '); %the datatype consists of floating point weight which can be used for splitting
l=length(tempNodes);
%fflag=1;
fromNode=M(str2num(tempNodes{1}));

if (l>1)
  %l=l-1; %matlab returns empty cell as well when splitting. this needs to be ignored
  
   for i=2:l
          toNode=M(str2num(tempNodes{i}));
           A(fromNode,toNode)=1;
  end

end

end
