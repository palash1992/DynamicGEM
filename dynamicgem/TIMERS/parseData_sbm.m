function A = parseData_sbm(filePath,M)
fileID=fopen(filePath);
fileLine=fgetl(fileID);
fileLine=fgetl(fileID);
fileLine=fgetl(fileID);
fileLine=fgetl(fileID); %first three line are metadata and needs to be ignored

A=sparse(double(M.Count),double(M.Count));

while ischar(fileLine)
 A=splitData_sbm(A,fileLine,M);
 fileLine=fgetl(fileID);
 end

end
