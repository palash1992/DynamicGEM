function TIMERS(dataFolder,K,Theta,datatype)
  
  %dataFolder='test_data'% directory with the test data consiting off numbers to denote time varying dynamic graph
 % time_slice=5% number of time slices to perform the embedding

  % Data
  	% A: an undirected N * N adjacency matrix
  	% E: an undirected M * 2 new edge sets

  %[A,E,T_Stamp] = Random_Com(5000,200000,0.3,0,8,100,0.9);
  totalFiles=dir([dataFolder '/*']);
  totalFiles(1:2,:)=[]; %matlab returns first two filenames as . and ..
  %totalTimeSlice=size(totalFiles,1);	
  time_slice=size(totalFiles,1)-1;  
  % Paramters
      % K: Embedding dimension
      % Theta: a threshold for re-run SVD
      % time_slice: divide new edges into equal timeslice
      % update: whether use updating


  %K = 128;                            
  %Theta = 0.17;
  %time_slice = 50;
  Update = 1;

  % Store all results
  U = cell(time_slice + 1,1);
  S = cell(time_slice + 1,1);
  V = cell(time_slice + 1,1);
  Loss_store = zeros(time_slice + 1,1);   % store loss for each time stamp
  Loss_bound = zeros(time_slice + 1,1);   % store loss bound for each time stamp
  run_times = 1;                          % store how many rerun times
  Run_t = zeros(time_slice + 1,1);        % store which timeslice re-run 

  % Calculate Original Similarity Matrix
  % In this reference implementation, we assumu similarity is adjacency matrix. Other variants shoule be straight-forward
  M=hashmapping_sbm([dataFolder '/0']);
  A=parseData_sbm([dataFolder '/0'],M); %get the first timeslice data for initialization
  Sim = A;
  N = size(A,1);

  % Calculate Static Solution
  [U{1},S{1},V{1}] = svds(Sim,K);
  U_cur = U{1} * sqrt(S{1});
  V_cur = V{1} * sqrt(S{1});

  dlmwrite(['output/' datatype '/0_U.txt'],U_cur,'delimiter',' ','precision',4);
  dlmwrite(['output/' datatype '/0_V.txt'],V_cur,'delimiter',' ','precision',4);


  Loss_store(1) = Obj(Sim, U_cur, V_cur);
  Loss_bound(1) = Loss_store(1);

  % Adding new edges
  %New_Edge_Num = length(E);
  % Store some useful variable
  S_cum = Sim;                  % store cumulated similarity matrix
  S_perturb = sparse(N,N);      % store cumulated perturbation from last rerun
  loss_rerun = Loss_store(1);   % store objective function of last rerun
  for i = 1:time_slice
  	% create the change in adjacency matrix
     % start_index = floor((i - 1) * New_Edge_Num / time_slice + 1);
     % end_index = floor(i * New_Edge_Num / time_slice);
     % if (size(E,2) == 2)       % if it is unweighted, unsigned
     %     A_add = sparse(E(start_index:end_index,1),E(start_index:end_index,2),1,N,N);
     % else                      % otherwise
     %     A_add = sparse(E(start_index:end_index,1),E(start_index:end_index,2),E(start_index:end_index,3),N,N);
     % end
     % A_add = A_add + A_add';   % assume each edge is undirected
      
      S_add = deltaA_sbm(S_cum, [dataFolder '/' num2str(i)],M); %hange to other functions for other similarities
      S_perturb = S_perturb + S_add;
      if (Update)
         % Some Updating Function Here
         [U{i+1},S{i+1},V{i+1}] = TRIP(U{i},S{i},V{i},S_add);
    
         % We use TRIP as an example, while other variants are permitted (as discussed in the paper)
         % Note that TRIP doesn't ensure smaller loss value
          U_cur = U{i+1} * sqrt(S{i+1});
          V_cur = V{i+1} * sqrt(S{i+1});
   
          dlmwrite(['output/' datatype '/incrementalSVD/' num2str(i) '_U.txt'],U_cur,'delimiter',' ','precision',4);
          dlmwrite(['output/' datatype '/incrementalSVD/' num2str(i) '_V.txt'],V_cur,'delimiter',' ','precision',4);

          
          Loss_store(i + 1) = Obj(S_cum + S_add, U_cur, V_cur);
      else
          Loss_store(i + 1) = Obj_SimChange(S_cum,S_add,U_cur,V_cur,Loss_store(i));
      end
      Loss_bound(i + 1) = RefineBound(Sim,S_perturb,loss_rerun,K);
      S_cum = S_cum + S_add;
      if (Loss_store(i + 1) >= (1 + Theta) * Loss_bound(i + 1))
          disp(['Begin rerun at time stamp:' num2str(i)]);
          Sim = S_cum;
          S_perturb = sparse(N,N);
          run_times = run_times + 1;
          Run_t(run_times) = i;
          [U{i+1},S{i+1},V{i+1}] = svds(Sim,K);
          U_cur = U{i+1} * sqrt(S{i+1});
          V_cur = V{i+1} * sqrt(S{i+1});

          %dlmwrite(['output/' datatype '/rerunSVD/' num2str(i) '_U.txt'],U_cur,'delimiter',' ','precision',4);
          %dlmwrite(['output/' datatype '/rerunSVD/' num2str(i) '_V.txt'],V_cur,'delimiter',' ','precision',4);

          loss_rerun = Obj(Sim,U_cur,V_cur);
          Loss_store(i + 1) = loss_rerun;
          Loss_bound(i + 1) = loss_rerun; 
      end
      dlmwrite(['output/' datatype '/rerunSVD/' num2str(i) '_U.txt'],U_cur,'delimiter',' ','precision',4);
      dlmwrite(['output/' datatype '/rerunSVD/' num2str(i) '_V.txt'],V_cur,'delimiter',' ','precision',4);
  end
  clear S_add S_cum S_perturb Sim;
  clear i start_index end_index loss_rerun;
  clear U_cur V_cur;

  % Evaluation
  Loss_optimal = zeros(time_slice + 1,1);   % store optimal result 
  Loss_optimal(1) = Loss_store(1);
  Sim = A;

  S_cum = Sim;
  for i = 1:time_slice
      S_add = getAddedEdge(S_cum,[dataFolder '/' num2str(i)],M);
      
      S_cum = S_cum + S_add; 
      [temp_U,temp_S,temp_V] = svds(S_cum,K);
      temp_U = temp_U * sqrt(temp_S);
      temp_V = temp_V * sqrt(temp_S);
      dlmwrite(['output/' datatype '/optimalSVD/' num2str(i) '_U.txt'],temp_U,'delimiter',' ','precision',4);
      dlmwrite(['output/' datatype '/optimalSVD/' num2str(i) '_V.txt'],temp_V,'delimiter',' ','precision',4);


      Loss_optimal(i + 1) = Obj(S_cum,temp_U,temp_V);
      disp(['Optimal ' num2str(i)]);
  end
  disp(max(Loss_store ./ Loss_optimal) - 1);

end