function M=hashmapping(filePath)

fileID=fopen(filePath);
fileLine=fgetl(fileID);
key=[];
val=[];
count=0;
while ischar(fileLine)
     
     temp =strsplit(fileLine,' ');
     node=str2num(temp{1});
     count=count+1;
     key=[key;node];
     val=[val; count] ;
     fileLine=fgetl(fileID);
end

M = containers.Map(key,val);

end
