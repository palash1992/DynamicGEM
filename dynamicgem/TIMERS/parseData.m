function A = parseData(filePath,M)
fileID=fopen(filePath);
fileLine=fgetl(fileID);
A=sparse(double(M.Count),double(M.Count));

while ischar(fileLine)
 A=splitData(A,fileLine,M);
 fileLine=fgetl(fileID);
 end

end
