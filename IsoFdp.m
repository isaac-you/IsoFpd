function [Clust,Coord,NMI,NMIv,ACCV,acc,adj_square,ObjV]=IsoFdp(netfile,dim1,dim2,metrics,NumCom1,NumCom2,trueLabel,nmi)
%input:
%    	netfile = the adjacency matrix with 2 columns: [nodeid1 nodeid2]
%		dim1 and dim2 = the range of the dimension
%		metrics = the method to calculate the similarity/distance:'structure' 'euclidean'... and so on
%		NumCom1 and NumCom2 = the range of the number of communities
%		trueLabel = the grandtruth community for LFR network
%		nmi = 1: the NMI of the result ;2: the ACC of the result
%output:
%		Clust = the community label of each node
%       Coord = the coordinates of the low dimension manifold for each node
%		NMI = the NMI of the result
%       NMIv = the NMI for each dimension
%		ACCV = the acc for each dimension
%		acc = the acc of the result
%		adj_square = the square form of input adjacency matrix
%		ObjV = the PD value for each dimension

%example : [~,~,resultnmi,~,~,~,~,~]=IsoFdp('LFR_data\\network_mix1.dat',10,30,'structure',15,50,'LFR_data\\community_mix1.dat',1);


tic;			 
x=load(netfile);
largex=x;       
nv=size(x,1);    
ND=nv/2;	       
n=max(x(:,2));	 
x(x(:,1)>x(:,2),:)=[];	


adj_square=sparse(n,n);
for i=1:size(x,1)
	mi=x(i,1);
	ni=x(i,2);
	adj_square(mi,ni)=1;
end
adj_square=adj_square+adj_square'+speye(n);

% 10 other mesures tested in the paper:'euclidean','cosine','jaccard'... and so on
if  ~strcmp(metrics,'structure')
	dist1=pdist(adj_square,metrics); 
	Diso=squareform(dist1);
end

% structure similarity 
t1=clock;		 
if  (strcmp(metrics,'structure'))
	SSfunc1=@(XI,XJ)(XI*XJ');     
	SSfunc2=@(XI,XJ)(sqrt(XI*XJ)); 
	intersectvector=sparse(pdist(adj_square,SSfunc1));
	MA=sum(adj_square,2);
	unionvector=sparse(pdist(MA,SSfunc2));
	SS=sparse(intersectvector./unionvector);
	VsimiMatrix=sparse(squareform(SS));
	%Diso=sparse(1)./VsimiMatrix;
	Diso=1./VsimiMatrix;
	Diso(logical(eye(n)))=0;
end

% get the nmi value or acc value for the community result
if nmi == 1  
   trueL=load(trueLabel);
   trueL=trueL(:,2);
   dimpool=dim1:dim2;
   NMIv(length(dimpool))=0;
   vertclustMatrix=zeros(length(dimpool),size(trueL,1));
   for i = 1:length(dimpool)
	   [vertclust,Coord,objv]=IsoCDP_Obj(adj_square,n,Diso,dimpool(i),NumCom1,NumCom2,'PD2');
	   NMIv(i)=NormalizedMI(trueL,vertclust');
	   ObjV(i)=objv;
	   vertclustMatrix(i,:)=vertclust;
	end
	[mobj,order1]=max(ObjV);
	NMI=NMIv(order1);
	Clust=vertclustMatrix(order1,:);
	ACCV=[];
	acc=[];
elseif nmi==2
	trueL=load(trueLabel);
   	trueL=trueL(:,2);
   	dimpool=dim1:dim2;
   	ACCV(length(dimpool))=0;
   	vertclustMatrix=zeros(length(dimpool),size(trueL,1));
   	for i = 1:length(dimpool)
	   [vertclust,Coord,objv]=IsoCDP_Obj(adj_square,n,Diso,dimpool(i),NumCom1,NumCom2,'PD2');
	   ACCV(i)=calculateAccuracy(trueL',vertclust);
	   ObjV(i)=objv;
	   vertclustMatrix(i,:)=vertclust;
	end
	[mobj,order1]=max(ObjV);
	acc=ACCV(order1);
	Clust=vertclustMatrix(order1,:);
	NMI=[];
	NMIv=[];
else
	dimpool=dim1:dim2;
	vertclustMatrix=cell(length(dimpool),1);
    for i = 1:length(dimpool)
		[vertclust,Coord,objv]=IsoCDP_Obj(adj_square,n,Diso,dimpool(i),NumCom1,NumCom2,'PD2');
		vertclustMatrix{i}=vertclust;
		ObjV(i)=objv;
	end
	[mobj,order1]=max(ObjV);
	Clust=vertclustMatrix{order1};
	NMI=[];
	NMIv=[];
	ACCV=[];
	acc=[];
end
toc;


function [vertclust,Coord,objv]=IsoCDP_Obj(adj_square,n,Diso,dim,NumCom1,NumCom2,objf)

	%================================IsomapII============================
	%calculate the low dimension coordinates for each node
	options.dims = 2:35;	 
	options.dijkstra=1;
	[Y]=IsomapII(Diso,'k', 100, options);
	Coordiso=Y.coords{dim};  
	Coord=Coordiso';

	%================================FdpI===============================
	% FdpI to calculate threes measures for each node
	xx=pdist(Coordiso','cosine');	 
	xx=[combnk(1:n,2),xx'];  
	ND=max(xx(:,2));
	NL=max(xx(:,1));
	if (NL>ND)
	  	ND=NL;
	end
	N=size(xx,1);
	for i=1:ND
	  	for j=1:ND
	    	dist(i,j)=0;
	  	end
	end
	for i=1:N
	  	ii=xx(i,1);
	  	jj=xx(i,2);
	  	dist(ii,jj)=xx(i,3);
	  	dist(jj,ii)=xx(i,3);
	end

	% calculate the local density for each node
	percent=2.0;
	fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent);
	position=round(N*percent/100);
	sda=sort(xx(:,3));
	dc=sda(position);
	fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);

	for i=1:ND
	  	rho(i)=0.;
	end
	for i=1:ND-1
	  	for j=i+1:ND
	    	if (dist(i,j)<dc)
	       	   rho(i)=rho(i)+1.;
	       	   rho(j)=rho(j)+1.;
	    	end
	  	end
	end

	% calculate the relative distance for each node
	maxd=max(max(dist));
	[rho_sorted,ordrho]=sort(rho,'descend');
	delta(ordrho(1))=-1.;
	nneigh(ordrho(1))=0;

	for ii=2:ND
	    delta(ordrho(ii))=maxd;
	    for jj=1:ii-1
	     	if (dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
	           delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
	           nneigh(ordrho(ii))=ordrho(jj);
	     	end
	    end
	end
	delta(ordrho(1))=max(delta(:));

	% calculate the third meauser for each node
	for i=1:ND
	  	ind(i)=i;
	  	gamma(i)=rho(i)*delta(i);
	end
	[gam_sorted,ordgam]=sort(gamma,'descend');
    
	
	%================================FdpII===============================
	% choose the number of communities and finish the assignment
	p(NumCom2-NumCom1+1)=0;		
	clustMatrix=zeros(NumCom2-NumCom1+1,ND);	
	for i = NumCom1:NumCom2    
		vertclust(1:ND)=-1; 	
		NCLUST=i;
		%choose the community center nodes
		temcenter=ordgam(1:i);
		for j = 1:i
			vertclust(temcenter(j))=j;
		end		

		%assigne the member nodes
		for im = 1:ND
	    	if (vertclust(ordrho(im))==-1)
	       	   vertclust(ordrho(im))=vertclust(nneigh(ordrho(im)));
	    	end
		end
	
		halo=vertclust;
	   	Modules=cell(1,length(unique(halo)));
	   	for iz = 1:length(unique(halo))
	   		Modules{iz}=find(halo==iz);
	   	end

		vertclustcell=cell(ND,1);
		for ik = 1:ND
			vertclustcell{ik}=halo(ik);
		end
		clustMatrix(i,:)=halo;
		%calculate the partition density 
		if strcmp(objf,'PD1')    
		   p(i)=PD1(adj_square,vertclustcell,i);
		elseif strcmp(objf,'PD2')  
		   p(i)=PD2(adj_square,vertclustcell,i);
		else
		   p(i)=modularity_metric(Modules,adj_square);	
		end	
	end
	[objv,NumCom_opt]=max(p);
	vertclust=clustMatrix(NumCom_opt,:);



%the original version of Isomap which is slow
function [Y, R, E] = Isomap(D, n_fcn, n_size, options);


N = size(D,1); 
if ~(N==size(D,2))
     error('D must be a square matrix'); 
end; 
if n_fcn=='k'
     K = n_size; 
     if ~(K==round(K))
         error('Number of neighbors for k method must be an integer');
     end
elseif n_fcn=='epsilon'
     epsilon = n_size; 
else 
     error('Neighborhood function must be either epsilon or k'); 
end
if nargin < 3
     error('Too few input arguments'); 
elseif nargin < 4
     options = struct('dims',1:10,'overlay',1,'comp',1,'display',1,'verbose',1); 
end
INF =  1000*max(max(D))*N;  %% effectively infinite distance

if ~isfield(options,'dims')
     options.dims = 1:10; 
end
if ~isfield(options,'overlay')
     options.overlay = 1;  
end
if ~isfield(options,'comp')
     options.comp = 1; 
end
if ~isfield(options,'display')
%     options.display = 1; 
     options.display = 0; 
end
if ~isfield(options,'verbose')
%     options.verbose = 1; 
     options.verbose = 0; 
end
dims = options.dims; 
comp = options.comp; 
overlay = options.overlay; 
displ = options.display; 
verbose = options.verbose; 

Y.coords = cell(length(dims),1); 
R = zeros(1,length(dims)); 

%%%%% Step 1: Construct neighborhood graph %%%%%
disp('Constructing neighborhood graph...'); 

if n_fcn == 'k'
     [tmp, ind] = sort(D); 
     for i=1:N
          D(i,ind((2+K):end,i)) = INF; 
     end
elseif n_fcn == 'epsilon'
     warning off    %% Next line causes an unnecessary warning, so turn it off
     D =  D./(D<=epsilon); 
     D = min(D,INF); 
     warning on
end

D = min(D,D');    %% Make sure distance matrix is symmetric

if (overlay == 1)
     E = int8(1-(D==INF));  %%  Edge information for subsequent graph overlay
end



%%%%% Step 2: Compute shortest paths %%%%%
disp('Computing shortest paths...'); 



tic; 
for k=1:N
     D = min(D,repmat(D(:,k),[1 N])+repmat(D(k,:),[N 1])); 
     if ((verbose == 1) && (rem(k,20) == 0)) 
          disp([' Iteration: ' num2str(k) 'Estimated time to completion:' num2str((N-k)*toc/k/60) 'minutes']); 
     end
end

%%%%% Remove outliers from graph %%%%%
disp('Checking for outliers...'); 
n_connect = sum(~(D==INF));        %% number of points each point connects to
[tmp, firsts] = min(D==INF);       %% first point each point connects to
[comps, I, J] = unique(firsts);    %% represent each connected component once
size_comps = n_connect(comps);     %% size of each connected component
[tmp, comp_order] = sort(size_comps);  %% sort connected components by size
comps = comps(comp_order(end:-1:1));    
size_comps = size_comps(comp_order(end:-1:1)); 
n_comps = length(comps);               %% number of connected components
if (comp>n_comps)                
     comp=1;                              %% default: use largest component
end
disp(['  Number of connected components in graph: ' num2str(n_comps)]); 
disp(['  Embedding component ' num2str(comp) ' with ' num2str(size_comps(comp)) ' points.']); 
Y.index = find(firsts==comps(comp)); 

D = D(Y.index, Y.index); 
N = length(Y.index); 

%%%%% Step 3: Construct low-dimensional embeddings (Classical MDS) %%%%%
disp('Constructing low-dimensional embeddings (Classical MDS)...'); 

opt.disp = 0; 
[vec, val] = eigs(-.5*(D.^2 - sum(D.^2)'*ones(1,N)/N - ones(N,1)*sum(D.^2)/N + sum(sum(D.^2))/(N^2)), max(dims), 'LR', opt); 

h = real(diag(val)); 
[foo,sorth] = sort(h);  sorth = sorth(end:-1:1); 
val = real(diag(val(sorth,sorth))); 
vec = vec(:,sorth); 

D = reshape(D,N^2,1); 
for di = 1:length(dims)
     if (dims(di)<=N)
         Y.coords{di} = real(vec(:,1:dims(di)).*(ones(N,1)*sqrt(val(1:dims(di)))'))'; 
         % r2 = 1-corrcoef(reshape(real(L2_distance(Y.coords{di}, Y.coords{di})),N^2,1),D).^2; 
         % R(di) = r2(2,1); 
         if (verbose == 1)
             disp(['  Isomap on ' num2str(N) ' points with dimensionality ' num2str(dims(di)) '  --> residual variance = ' num2str(R(di))]); 
         end
     end
end

clear D; 

%%%%%%%%%%%%%%%%%% Graphics %%%%%%%%%%%%%%%%%%

if (displ==1)
     %%%%% Plot fall-off of residual variance with dimensionality %%%%%
     figure;
     hold on
     plot(dims, R, 'bo'); 
     plot(dims, R, 'b-'); 
     hold off
     ylabel('Residual variance'); 
     xlabel('Isomap dimensionality'); 

     %%%%% Plot two-dimensional configuration %%%%%
     twod = find(dims==2); 
     if ~isempty(twod)
         figure;
         hold on;
         plot(Y.coords{twod}(1,:), Y.coords{twod}(2,:), 'ro'); 
         if (overlay == 1)
             gplot(E(Y.index, Y.index), [Y.coords{twod}(1,:); Y.coords{twod}(2,:)]'); 
             title('Two-dimensional Isomap embedding (with neighborhood graph).'); 
         else
             title('Two-dimensional Isomap.'); 
         end
         hold off;
     end
end

return;


%PD1、PD2分别是原始PD和张老师PD
function P = PD1(A, label, k)
% get the partition density of the given graph partition.
% A: adjacency matrix.
% label: each cell is the community label of each node.
% k: the community number
% P: partition density

if A(1,1) == 1
    A = A-eye(size(A));		%把邻接方阵主对角线元素改为0
end

N = 0;
M = sum(sum(A))/2;  % number of edges.
pool = cell(k,1);  % each cell is one community containing node labels.
s = zeros(length(A),1); % each element is the label number of the node.

for i = 1:size(label,1)
    if (~isempty(label{i}))&&(label{i}~=0)	%如果节点i不是噪声点
        temp = label{i};	%暂时把节点i的类标签存储为temp
        s(i) = length(temp);	%得到该节点归属类簇的个数
        for j = 1:length(temp)
            pool{temp(j)} = union([pool{temp(j)}],i);
        end
    else
        N = N+1;
    end
end
N = N+sum(s);

n = zeros(1,k); % number of nodes in each community
m = zeros(1, k); % number of edges in each community
d = zeros(1,k); % link density of each community
n2 = zeros(1,k); % max label number in each community
for t = 1:k
    n(t) = length(pool{t});
    if ~isempty(s(pool{t}))
        n2(t) = max(s(pool{t}));
    end
    m(t) = sum(sum(A(pool{t}, pool{t})))/2;
    if (n(t) > 2)
        d(t) = m(t)*(m(t)-(n(t)-1))/((n(t)*(n(t)-1)/2)-(n(t)-1));
    end
end

%P = 2*sum(d)/M;
P = sum(d)/(M*(k^0.5));
%================================================
function P = PD2(A, label, k)
% get the partition density of the given graph partition.
% A: adjacency matrix.
% label: each cell is the community label of each node.
% k: the community number
% P: partition density

if A(1,1) == 1
    A = A-eye(size(A));		%把邻接方阵主对角线元素改为0
end

N = 0;
M = sum(sum(A))/2;  % number of edges.
pool = cell(k,1);  % each cell is one community containing node labels.
s = zeros(length(A),1); % each element is the label number of the node.

for i = 1:size(label,1)
    if (~isempty(label{i}))&&(label{i}~=0)	%如果节点i不是噪声点
        temp = label{i};	%暂时把节点i的类标签存储为temp
        s(i) = length(temp);	%得到该节点归属类簇的个数
        for j = 1:length(temp)
            pool{temp(j)} = union([pool{temp(j)}],i);
        end
    else
        N = N+1;
    end
end
N = N+sum(s);

n = zeros(1,k); % number of nodes in each community
m = zeros(1, k); % number of edges in each community
d = zeros(1,k); % link density of each community
n2 = zeros(1,k); % max label number in each community
for t = 1:k
    n(t) = length(pool{t});
    if ~isempty(s(pool{t}))
        n2(t) = max(s(pool{t}));
    end
    m(t) = sum(sum(A(pool{t}, pool{t})))/2;
    if (n(t) > 2)
        d(t) = (n(t)/n2(t))*(m(t)-(n(t)-1))/((n(t)*(n(t)-1)/2)-(n(t)-1));
    end
end

%P = sum(d)/N;
P = sum(d)/(N*(k^0.5));






function Q=modularity_metric(modules,adj)

nedges=numedges(adj); % total number of edges

Q = 0;
for m=1:length(modules)

  e_mm=numedges(adj(modules{m},modules{m}))/nedges;
  a_m=sum(sum(adj(modules{m},:)))/(2*nedges);
  Q = Q + (e_mm - a_m^2);
  
end




function m = numedges(adj)

sl=selfloops(adj); % counting the number of self-loops

if issymmetric(adj) & sl==0    % undirected simple graph
    m=sum(sum(adj))/2; 
    return
elseif issymmetric(adj) & sl>0
    sl=selfloops(adj);
    m=(sum(sum(adj))-sl)/2+sl; % counting the self-loops only once
    return
elseif not(issymmetric(adj))   % directed graph (not necessarily simple)
    m=sum(sum(adj));
    return
end


% Checks whether a matrix is symmetric (has to be square)
% Check whether mat=mat^T
% INPUTS: adjacency matrix
% OUTPUTS: boolean variable, {0,1}
% GB, October 1, 2009

function S = issymmetric(mat)

S = false; % default
if mat == transpose(mat); S = true; end


% counts the number of self-loops in the graph
% INPUT: adjacency matrix
% OUTPUT: interger, number of self-loops
% Last Updated: GB, October 1, 2009

function sl=selfloops(adj)

sl=sum(diag(adj));




%calculate NMI value
function [NMI] = NormalizedMI(trueLabel, partitionMatrix)
% normalized mutual information
% Author: Weike Pan, weikep@cse.ust.hk
% Ref: Dhilon, KDD 2004 Kernel k-means, Spectral Clustering and Normalized Cuts
% Section 6.3
% High NMI value indicates that the clustering and true labels match well 

% usage: NormalizedMI([1 1 1 2 2 2]', [1 2 1 3 3 3]')

%%
truey = trueLabel;
[m1, c] = size(truey); % c: class #

PM = partitionMatrix;
[m2, k] = size(PM); % k: cluster #

%%
% check whether m1 == m2
if m1 ~= m2
    error('m1 not equal m2');    
else
    m = m1;
end

%% change the truelable or the partition matrix: m \times c
if c == 1
    c = length( unique(truey) );
    tmp = zeros(m,c);
    for i = 1 : c
        tmp((truey == i), i) = 1;
    end
    truey = tmp;    
end

if k == 1
    k = length( unique(PM) );
    tmp = zeros(m,k);
    for i = 1 : k
        tmp((PM == i), i) = 1;
    end
    PM = tmp;    
end

%%

% *****************************
% calculate the confusion matrix
for l = 1 : 1 : k  
    for h = 1 : 1 : c
        n(l,h) = sum( (truey(:,h) == 1) & (PM(:,l) == 1) );    
    end
end



% *****************************
NMI = 0;
for l = 1 : 1 : k
    
    for h = 1 : 1 : c
        NMI = NMI + (n(l,h)/m) * log(  ( n(l,h)*m + eps) / ( sum(n(:,h))*sum(n(l,:)) + eps) ); 
    end

end

Hpi = - sum( (sum(PM)/m) .* log( sum(PM)/m + eps ) );
Hvarsigma = - sum( (sum(truey)/m) .* log( sum(truey)/m + eps ) );

% NMI = 2*NMI/(Hpi + Hvarsigma);

% JMLR03, A. Strehl and J. Ghosh. Cluster ensembles -- a knowledge reuse framework for combining multiple partitions.
NMI = NMI/sqrt(Hpi*Hvarsigma);

%Calulate acc value this version use perms and cannot handle big networks
function [Acc,rand_index,match]=AccMeasure(T,idx)
%Measure percentage of Accuracy and the Rand index of clustering results
% The number of class must equal to the number cluster 

%Output
% Acc = Accuracy of clustering results
% rand_index = Rand's Index,  measure an agreement of the clustering results
% match = 2xk mxtrix which are the best match of the Target and clustering results

%Input
% T = 1xn target index
% idx =1xn matrix of the clustering results

% EX:
% X=[randn(200,2);randn(200,2)+6,;[randn(200,1)+12,randn(200,1)]]; T=[ones(200,1);ones(200,1).*2;ones(200,1).*3];
% idx=kmeans(X,3,'emptyaction','singleton','Replicates',5);
%  [Acc,rand_index,match]=Acc_measure(T,idx)

k=max(T);
n=length(T);
for i=1:k
    temp=find(T==i);
    a{i}=temp; %#ok<AGROW>
end

b1=[];
t1=zeros(1,k);
for i=1:k
    tt1=find(idx==i);
    for j=1:k
       t1(j)=sum(ismember(tt1,a{j}));
    end
    b1=[b1;t1]; %#ok<AGROW>
end
    Members=zeros(1,k); 
    
P = perms((1:k));
    Acc1=0;
for pi=1:size(P,1)
    for ki=1:k
        Members(ki)=b1(P(pi,ki),ki);
    end
    if sum(Members)>Acc1
        match=P(pi,:);
        Acc1=sum(Members);
    end
end

rand_ss1=0;
rand_dd1=0;
for xi=1:n-1
    for xj=xi+1:n
        rand_ss1=rand_ss1+((idx(xi)==idx(xj))&&(T(xi)==T(xj)));
        rand_dd1=rand_dd1+((idx(xi)~=idx(xj))&&(T(xi)~=T(xj)));
    end
end
rand_index=200*(rand_ss1+rand_dd1)/(n*(n-1));
Acc=Acc1/n*100; 
match=[1:k;match];

%Calulate acc value this version  can handle big networks
function [micro_precision]=calculateAccuracy(true_labels,Ensemble_labels)
%
% Calculate  micro-precision given clustering results and true labels.
%
%   k = number of ensemble clusters
%   M = number of data points
%
% Input:
%   true_labels:        1*M, true class labels for the data points
%   Ensemble_labels:    1*M, labels obtained from BCE
    
% Output:
%   micro_precision:    micro-precision
%--------------------------------------------------------------------

k=length(unique(true_labels));
M=length(true_labels);
   
    for j=1:k
         for jj=1:k
            [xx,accurence(j,jj)]=size(find(((Ensemble_labels==jj)*j)==true_labels));
         end
    end 

    [rowm,coln]=size(accurence);
    amatrix=accurence;
    sumMax=0;
    while rowm>=1
        xx=max(max(amatrix));
        [x,y]=find(amatrix==xx);
        sumMax=sumMax+xx;                      
        iyy=1;
        temp=zeros(rowm,rowm-1);
        for iy=1:rowm
            if iy==y(1)
                continue;
            else                        
                temp(:,iyy)=amatrix(:,iy);
                iyy=iyy+1;
             end
         end  
         temp2=zeros(rowm-1,rowm-1);
         ixx=1;
         for ix=1:rowm
            if ix==x(1)
                continue;
            else                        
                temp2(ixx,:)=temp(ix,:);
                ixx=ixx+1;
             end
          end
          rowm=rowm-1;
          amatrix=zeros(rowm,rowm);
          amatrix=temp2;
                  
    end

   micro_precision=sumMax/M;