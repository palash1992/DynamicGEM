function A=splitData(A,fileLine,M)
vals=strsplit(fileLine,'.0'); %the datatype consists of floating point weight which can be used for splitting
l=length(vals);
fflag=1;
fromNode=0;
if (l>1)
  l=l-1; %matlab returns empty cell as well when splitting. this needs to be ignored
  for i=1:l
        tempNodes=strsplit(vals{i},' ');
        if fflag
          fromNode=M(str2num(tempNodes{1}));
          toNode=M(str2num(tempNodes{2}));
          weight=str2num(tempNodes{3});
          fflag=0;
        else
          toNode=M(str2num(tempNodes{2}));
          weight=str2num(tempNodes{3});
        end
        A(fromNode,toNode)=weight;
  end

end

end
