function [Clust,Coord,NMI,NMIv,ACCV,acc,adj_square,ObjV]=IsoFdp(netfile,dim1,dim2,metrics,NumCom1,NumCom2,trueLabel,objf,nmi)

%本算法首先利用isomap对网络两两相似度进行降纬，再计算降纬后坐标的两两距离
%在此距离基础上利用cluster_dp算法聚类  此本版集合了'strength'节点距离测度 2015-04-17
%此版本集合了结构相似度'structure'来自<Novel heuristic density-based method for community
%detection in networks>(2014) 并且 利用PD函数实现自动化聚类 2015-04-20
%netfile='长方邻接矩阵'  NumCom表示类迭代次数(2:NumCom)仿造SBMF(2013) 边聚类暂时没有vc=0时取[]
%type='Extend'or'Jaccard'分别表示两种边相似度测度 vc=1时取[]
%dim表示在iso某纬度中提取坐标
%vc=1节点聚类  vc=0边聚类
%metrics='diffusionkernel' 'strength' 或者其他pdist自动测度 当vc=1时选择距离测度 vc=0取[]
%此版本集合了PD和modularity两个目标函数实现自动化聚类 2014-04-21 
%obj='PD' or 'anythingelse' PD 
%'Extend'来自<Link Clustering with Extended Link Similarity and EQ Evaluation Division>(2013)提供的广义相似度公式
%'Jaccard'来自<Link communities reveal multiscale complexity in networks>(2010)提供的Jaccard相似度公式
%'strength'来自<VISUAL CLUSTERING OF COMPLEX NETWORK BASED ON NONLINEAR DIMENSION REDUCTION>(2006)引用的边强度倒数作为节点距离测度
%此版本用于测试LFR非重叠数据集,2015-05-25,由于该算法是确定性算法且速度比较慢,所以建议只算1次NMI
%示例：
%>> [vertclust,Linkclust,NMI]=IsoCdp_NMI_LFR('network_06.dat',[],25,1,'structure',50,'community_06.dat','q');
%为了在network_06数据集上取得好结果,此版本注释了352-356,屏蔽噪声

%此版本可以探测GN网络/LFR(非重叠)网络/以及真实网络(足球/海豚/jazz...) 2015-06-16
%示例：人工网络需要NMI则输入参数nmi=1 
%2015-08-06: 输出降维后的坐标矩阵 以供DBSCAN kmeans聚类
%>> [vertclust,Coord,NMI04375]=IsoFdp_Com('network_0.4375.dat',[],4,1,'structure',50,'community_0.4375.dat','PD',1);
%示例：真实网络不需要NMI则输入参数nmi=0
%>> [vertclustjaz,Coord,nmi,objjaz]=IsoFdp_Com('jazzed.txt',[],6,1,'structure',50,[],'PD',0);
%此版本用以测试dim敏感性/同时添加原始PD/真实网络绘图 2015-10-19
%添加参数dim1,dim2用以控制维度范围
%NumCom1,NumCom2控制类簇个数从哪到哪
%集合维度在内一起找最大PD得到最终聚类结果
% [Clust,Coord,NMI,adj_square,ObjV]=IsoFdp_Com('network_1_zout7.dat',[],2,25,1,'structure',2,6,'community_1_zout7.dat','PD2',1);
tic;			 
x=load(netfile);
largex=x;       
nv=size(x,1);     %得到2列格式邻接矩阵的行数(网络边数的2倍)
ND=nv/2;	         %网络边数
n=max(x(:,2));	 %节点数
x(x(:,1)>x(:,2),:)=[];	%无重复长方邻接阵


%   net=load(netfile);      %%针对无权无向网络，且每个节点至少有一条边(不适用于有隔离节点的矩阵)
%   nv=size(net,1);         %%得到2/3列格式邻接矩阵行数(网络边数2倍)
%   ND=length(unique(net(:,1)));  %%得到网络总节点数
%   adj_square=zeros(ND);         %%得到ND*ND的0方阵
adj_square=zeros(n);          %%得到n*n的0方阵
%%如果邻接阵只有2列，则为无权网络
%%返回二值邻接方阵(无权网络)
if size(x,2)<3          
	for i=1:ND
		ii=x(i,1);             
		jj=x(i,2);        
		adj_square(ii,jj)=1;     %得到邻接方阵
		adj_square(jj,ii)=1;
	end
%%返回double邻接方阵(有权网络)

elseif size(net,2)==3   
	for i=1:ND                  
		ii=x(i,1);             
		jj=x(i,2);        
		adj_square(ii,jj)=x(i,3);       
		adj_square(jj,ii)=x(i,3);
	end
end

%%如果没有指定特定距离则默认使用欧式距离
%%得到一个ND*(ND-1)/2的行向量,把邻接方阵当作n纬空间n个点的坐标矩阵求得此两两距离  
%%如果指定了测度但不是'diffusionkernel'那么将该指定测度传递给pdist     
if  all([(~strcmp(metrics,'diffusionkernel')),(~strcmp(metrics,'strength')),(~strcmp(metrics,'structure'))])  
	dist1=pdist(adj_square,metrics); 
	Diso=squareform(dist1);
end

%%如果输入的第二参数指定为'diffusionkernel'则单独计算距离矩阵
%%计算时要求输入beta值
%%得到的初始距离阵S为方阵，
if  (strcmp(metrics,'diffusionkernel'))
	disp('need the beta for diffusionkernel') 
	beta=input('value of beta for diffusionkernel \n');
	z=adj_square;

	for i = 1:n
		z(i,i)=-sum(adj_square(i,:),2);
	end

	temp0 = beta*z;
	K = expm(temp0);
	D = diag(K);
	S = K./sqrt((repmat(D,1,n).*repmat(D',n,1))); 
%%将距离方阵S转换为n*(n-1)/2距离行向量dist1
	k=0;
	for i=1:n-1
		for j=i+1:n
	   		k=k+1;
	   		dist1(k)=S(i,j); 
		end  
	end
	Diso=squareform(dist1);
end

if  (strcmp(metrics,'strength'))
%		x=load(netfile);
%		largex=x;       
%		nv=size(x,1);          %得到2列格式邻接矩阵的行数(网络边数的2倍)
%		ND=nv/2;			   %网络边数
%		n=max(x(:,2));			   %节点数
%		adj_square=zeros(n);   %初始化邻接方阵
%		for i=1:nv			   %长方邻接阵转换为邻接方阵
%			ii=x(i,1);         %有重复边2列长方邻接阵 也适用于无重复边2列长方阵      
%			jj=x(i,2);       
%			adj_square(ii,jj)=1;   %邻接方阵为以后取子图方便  
%			adj_square(jj,ii)=1;
%		end
%		x(x(:,1)>x(:,2),:)=[];

	%neighbour=cell(n,1);	%初始化n个节点领域数组
	%for i = 1:n
	%	neighbour{i}=[i;largex(largex(:,1)==i,2)];  %得到所有点领域数组
	%end 


	%本算法每条边对应点的领域是相对变化的 当然特定两点公共领域点不会变
	%边相似度算法中点领域是固定的
	%因此需要根据每条边分别计算左右两点的排他领域和公共领域
	Affinity.left=cell(ND,1);	%初始化每条边左边点的排他领域
	Affinity.right=cell(ND,1);	%初始化每条边右边点的排他领域
	Affinity.common=cell(ND,1);	%初始化每条边两点的公共领域

	exneighbour=cell(n,1);  %初始化n个节点领域数组
	for i = 1:n
	  	exneighbour{i}=largex(largex(:,1)==i,2);  %得到所有点领域数组(没有自己)
	end 

	%intercatrix=cell(n,n); 	%初始化节点两两交集元胞(不包含自己的公共领域点)
	%for i = 1:n-1
	%	for j =i+1:n
	%		intercatrix{i,j}=intersect(exneighbour(i),exneighbour(j));
	%	end
	%end

	for i = 1:ND
	%	leftwhole=largex(largex(:,1)==x(i,1)),2);   %第i条边左边点全部领域点(没有自己)
	%	rightwhole=largex(largex(:,1)==x(i,2)),2);	%第i条边右边点全部领域点(没有自己)
	%	common=intersect(exneighbour{x(i,1)},exneighbour{x(i,2)}); %第i条边两点公共领域点(没有自己)
	%	left=setdiff(exneighbour{x(i,1)},[exneighbour{x(i,2)},x(i,2)]);  %左边点排他领域(没有自己)
	%	right=setdiff(exneighbour{x(i,2)},[exneighbour{x(i,1)},x(i,1)]); %右边点排他领域(没有自己)
	%	Affinity.left{i}=left;
		Affinity.left{i}=(setdiff(exneighbour{x(i,1)},[exneighbour{x(i,2)};x(i,2)]))';
	%	Affinity.right{i}=right;
		Affinity.right{i}=(setdiff(exneighbour{x(i,2)},[exneighbour{x(i,1)};x(i,1)]))';
	%	Affinity.common{i}=common;
		Affinity.common{i}=(intersect(exneighbour{x(i,1)},exneighbour{x(i,2)}))';
	end

	strength=zeros(1,ND);	%初始化边强度向量

	for i = 1:ND  	%求得每条边左右两点排他子图  公共领域子图
	%	subleft=adj_square(Affinity.left{i},Affinity.left{i});    %左边点排他子图
	%	subright=adj_square(Affinity.right{i},Affinity.right{i}); %右边点排他子图
		subcommon=adj_square(Affinity.common{i},Affinity.common{i}); %公共领域子图
		LinkCC=sum(sum(subcommon)); 	%公共领域子图内部连边数2倍
		linkLR=sum(sum(adj_square(Affinity.left{i},Affinity.right{i})))/2;  %两个节点排他子图之间连边数
		LinkLC=sum(sum(adj_square(Affinity.left{i},Affinity.common{i})))/2; %左边排他子图和公共子图之间连边数
		LinkRC=sum(sum(adj_square(Affinity.right{i},Affinity.common{i})))/2;%右边排他子图和公共子图之间连边数
		SLR=linkLR/(length(Affinity.left{i})*length(Affinity.left{i}));     %左右两个排他领域强度
		if isnan(SLR)
		   SLR=0;	
		end
		SLC=LinkLC/(length(Affinity.left{i})*length(Affinity.common{i}));	
		if isnan(SLC)
		   SLC=0;	
		end
		SRC=LinkRC/(length(Affinity.right{i})*length(Affinity.common{i})); 
		if isnan(SRC)
		   SRC=0;	
		end 
		SCC=LinkCC/(length(Affinity.common{i})*(length(Affinity.common{i})-1)); %公共领域子图内部强度
		if isnan(SCC)
		   SCC=0;
		end  
		pro3len=length(Affinity.common{i})/(length(Affinity.left{i})+length(Affinity.right{i})+length(Affinity.common{i}));
		if isnan(pro3len)
		   pro3len=0;
		end  
		strength(i)=SLR+SLC+SRC+SCC+pro3len;  	%最终连边强度
	end

	Diso11=1./zeros(n);	   %初始化距离矩方阵 默认为Inf
	for i=1:ND			   %利用无重复邻接长阵x
		ii=x(i,1);              
		jj=x(i,2);       
		Diso11(ii,jj)=1/strength(i);   %两点距离等于其连边强度倒数 
		Diso11(jj,ii)=1/strength(i);
	end
	Diso11(logical(eye(size(Diso11,1))))=0;	%主对角线节点自己距离为0
	Diso=Diso11;
end	

if  (strcmp(metrics,'structure'))
	neighbour=cell(n,1);  %初始化n个节点领域数组
	VsimiMatrix=zeros(n); %初始化节点结构相似度矩阵
	for i = 1:n
	  	neighbour{i}=[i;largex(largex(:,1)==i,2)];  %得到所有点领域数组(包括自己)
	end 
	for i = 1:n-1
		for j =i+1:n
			VsimiMatrix(i,j)=length(intersect(neighbour{i},neighbour{j}))/sqrt(length(neighbour{i})*length(neighbour{j}));
		end
	end
	VsimiMatrix=VsimiMatrix+VsimiMatrix'+eye(n);
	Diso=1./VsimiMatrix;
	Diso(logical(eye(size(Diso,1))))=0;
end


if nmi == 1
   trueL=load(trueLabel);
   trueL=trueL(:,2);

   %for i = 1:10
   dimpool=dim1:dim2;
   NMIv(length(dimpool))=0;
   vertclustMatrix=zeros(length(dimpool),size(trueL,1));
   for i = 1:length(dimpool)
	   [vertclust,Coord,objv]=IsoCDP_Obj(adj_square,n,Diso,dimpool(i),NumCom1,NumCom2,objf);
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
	   [vertclust,Coord,objv]=IsoCDP_Obj(adj_square,n,Diso,dimpool(i),NumCom1,NumCom2,objf);
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
		[vertclust,Coord,objv]=IsoCDP_Obj(adj_square,n,Diso,dimpool(i),NumCom1,NumCom2,objf);
		vertclustMatrix{i}=vertclust;
		ObjV(i)=objv;
	end
	[mobj,order1]=max(ObjV);
	Clust=vertclustMatrix{order1};
	NMI=[];
	NMIv=[];
end
toc;

%function [vertclust,Coord,objv]=IsoCDP_Obj(adj_square,n,Diso,dim,NumCom1,NumCom2,objf,isfast)
function [vertclust,Coord,objv]=IsoCDP_Obj(adj_square,n,Diso,dim,NumCom1,NumCom2,objf)


	 options.dims = 2:35;	 %降纬的目标空间
	
	% if isfast==1
		%options.landmarks = 1:(0.5*n);	 %landmark iso选中的mark点数   0.963612304522893  0.9659  0.9610
		% if n>500
		% 	options.landmarks = 1:(0.25*n);
		% end	
		options.dijkstra=1;
		[Y]=IsomapII(Diso,'k', 100, options);
	% else
	% 	[Y,~,~]=Isomap(Diso,'k', 100, options);
	% end
		%options.verbose=0;      %不输出中间过程
		%options.display=0;		 %不出2纬映射图
		%options.overlay=0;		 %不出残差方差图
	

	Coordiso=Y.coords{dim};  %选择iso生成的dim纬度坐标 计算下一步距离矩阵
	Coord=Coordiso';
	xx=pdist(Coordiso','cosine');	 %得到iso生成坐标的两两距离向量
%	xx=pdist(Coordiso','cityblock');	 %得到iso生成坐标的两两距离向量
%	xx=pdist(Coordiso','hamming');
	xx=[combnk(1:n,2),xx'];  %生成clust_dp需要的三列格式距离矩阵

	%============================================================
	%iso低纬坐标的三列格式距离矩阵,利用cluster_dp对其聚类
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
	percent=2.0;
%	percent=2.5;
%	percent=3.0;
%	percent=3.5;
%	percent=4.0;
%	percent=4.5;
%	percent=5.0;
	fprintf('average percentage of neighbours (hard coded): %5.6f\n', percent);

	position=round(N*percent/100);
	sda=sort(xx(:,3));
	dc=sda(position);

	fprintf('Computing Rho with gaussian kernel of radius: %12.6f\n', dc);


	for i=1:ND
	  	rho(i)=0.;
	end
	%
	% Gaussian kernel
	%
	%for i=1:ND-1
	%  for j=i+1:ND
	%     rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
	%     rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
	%  end
	%end
	%
	% "Cut off" kernel
	%
	for i=1:ND-1
	  	for j=i+1:ND
	    	if (dist(i,j)<dc)
	       	   rho(i)=rho(i)+1.;
	       	   rho(j)=rho(j)+1.;
	    	end
	  	end
	end

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
%	disp('Generated file:DECISION GRAPH')
%	disp('column 1:Density')
%	disp('column 2:Delta')

%	fid = fopen('DECISION_GRAPH', 'w');
%	for i=1:ND
%	   	fprintf(fid, '%6.2f %6.2f\n', rho(i),delta(i));
%	end

%	disp('Select a rectangle enclosing cluster centers')
%	scrsz = get(0,'ScreenSize');
%	figure('Position',[6 72 scrsz(3)/4. scrsz(4)/1.3]);
	for i=1:ND
	  	ind(i)=i;
	  	gamma(i)=rho(i)*delta(i);
	end

	[gam_sorted,ordgam]=sort(gamma,'descend');
    
	
	p(NumCom2-NumCom1+1)=0;		%初始化p值向量
    
	clustMatrix=zeros(NumCom2-NumCom1+1,ND);	%初始化每个类数下的聚类结果储存矩阵	
	for i = NumCom1:NumCom2    
		vertclust(1:ND)=-1; 	%初始化节点类标签向量
		NCLUST=i;
		temcenter=ordgam(1:i);
		for j = 1:i
			vertclust(temcenter(j))=j;
		end			
		%assignation
		for im = 1:ND
	    	if (vertclust(ordrho(im))==-1)
	       	   vertclust(ordrho(im))=vertclust(nneigh(ordrho(im)));
	    	end
		end
		%halo
		halo=vertclust;
		% for imm=1:i
	 %       	bord_rho(i)=0.;
	 %    end
	 %   	for itt = 1:ND-1
	 %    	for jtt = itt+1:ND
	 %      		if ((vertclust(itt)~=vertclust(jtt))&&(dist(itt,jtt)<=dc))
	 %        		rho_aver=(rho(itt)+rho(jtt))/2.;
	 %        		if (rho_aver>bord_rho(vertclust(itt)))
	 %          			bord_rho(vertclust(itt))=rho_aver;
	 %        		end
	 %        		if (rho_aver>bord_rho(vertclust(jtt)))
	 %          			bord_rho(vertclust(jtt))=rho_aver;
	 %        		end
	 %      		end
	 %    	end
	 %   	end
%	   	for inn = 1:ND
%	    	if (rho(inn)<bord_rho(vertclust(inn)))
%	      		halo(inn)=0;
%	    	end
%	   	end

	   	Modules=cell(1,length(unique(halo)));
	   	for iz = 1:length(unique(halo))
	   		Modules{iz}=find(halo==iz);
	   	end

		vertclustcell=cell(ND,1);
		for ik = 1:ND
			vertclustcell{ik}=halo(ik);
		end
		clustMatrix(i,:)=halo;
		if strcmp(objf,'PD1')    %原始PD
		   p(i)=PD1(adj_square,vertclustcell,i);
		elseif strcmp(objf,'PD2')  %张老师PD
		   p(i)=PD2(adj_square,vertclustcell,i);
		else
		   p(i)=modularity_metric(Modules,adj_square);	
		end	
	end
	[objv,NumCom_opt]=max(p);
	vertclust=clustMatrix(NumCom_opt,:);



function [Y, R, E] = Isomap(D, n_fcn, n_size, options);

% ISOMAP   Computes Isomap embedding using the algorithm of 
%             Tenenbaum, de Silva, and Langford (2000). 
%
% [Y, R, E] = isomap(D, n_fcn, n_size, options); 
%
% Input:
%    D = N x N matrix of distances (where N is the number of data points)
%    n_fcn = neighborhood function ('epsilon' or 'k') 
%    n_size = neighborhood size (value for epsilon or k) 
%
%    options.dims = (row) vector of embedding dimensionalities to use
%                        (1:10 = default)
%    options.comp = which connected component to embed, if more than one. 
%                        (1 = largest (default), 2 = second largest, ...)
%    options.display = plot residual variance and 2-D embedding?
%                        (1 = yes (default), 0 = no)
%    options.overlay = overlay graph on 2-D embedding?  
%                        (1 = yes (default), 0 = no)
%    options.verbose = display progress reports? 
%                        (1 = yes (default), 0 = no)
%
% Output: 
%    Y = Y.coords is a cell array, with coordinates for d-dimensional embeddings
%         in Y.coords{d}.  Y.index contains the indices of the points embedded.
%    R = residual variances for embeddings in Y
%    E = edge matrix for neighborhood graph
%

%    BEGIN COPYRIGHT NOTICE
%
%    Isomap code -- (c) 1998-2000 Josh Tenenbaum
%
%    This code is provided as is, with no guarantees except that 
%    bugs are almost surely present.  Published reports of research 
%    using this code (or a modified version) should cite the 
%    article that describes the algorithm: 
%
%      J. B. Tenenbaum, V. de Silva, J. C. Langford (2000).  A global
%      geometric framework for nonlinear dimensionality reduction.  
%      Science 290 (5500): 2319-2323, 22 December 2000.  
%
%    Comments and bug reports are welcome.  Email to jbt@psych.stanford.edu. 
%    I would also appreciate hearing about how you used this code, 
%    improvements that you have made to it, or translations into other
%    languages.    
%
%    You are free to modify, extend or distribute this code, as long 
%    as this copyright notice is included whole and unchanged.  
%
%    END COPYRIGHT NOTICE


%%%%% Step 0: Initialization and Parameters %%%%%

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

% Finite entries in D now correspond to distances between neighboring points. 
% Infinite entries (really, equal to INF) in D now correspond to 
%   non-neighoring points. 

%%%%% Step 2: Compute shortest paths %%%%%
disp('Computing shortest paths...'); 

% We use Floyd's algorithm, which produces the best performance in Matlab. 
% Dijkstra's algorithm is significantly more efficient for sparse graphs, 
% but requires for-loops that are very slow to run in Matlab.  A significantly 
% faster implementation of Isomap that calls a MEX file for Dijkstra's 
% algorithm can be found in isomap2.m (and the accompanying files
% dijkstra.c and dijkstra.dll). 

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




% Computing the modularity for a given module/commnunity break-down
% Defined as: Q=sum_over_modules_i (eii-ai^2) (eq 5) in Newman and Girvan.
% eij = fraction of edges that connect community i to community j, ai=sum_j (eij)
% Source: Newman, M.E.J., Girvan, M., "Finding and evaluating community structure in networks"
% Also: "Fast algorithm for detecting community structure in networks", Mark Newman
% Inputs: adjacency matrix and set modules as cell array of vectors, ex: {[1,2,3],[4,5,6]}
% Outputs: modularity metric, in [-1,1]
% Other functions used: numedges.m
% Last updated: June 13, 2011

function Q=modularity_metric(modules,adj)

nedges=numedges(adj); % total number of edges

Q = 0;
for m=1:length(modules)

  e_mm=numedges(adj(modules{m},modules{m}))/nedges;
  a_m=sum(sum(adj(modules{m},:)))/(2*nedges);
  Q = Q + (e_mm - a_m^2);
  
end



% Returns the total number of edges given the adjacency matrix
% Valid for both directed and undirected, simple or general graph
% INPUTs: adjacency matrix
% OUTPUTs: m - total number of edges/links
% Other routines used: selfloops.m, issymmetric.m
% GB, Last Updated: October 1, 2009

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


% function d = L2_distance(a,b,df)
% % L2_DISTANCE - computes Euclidean distance matrix
% %
% % E = L2_distance(A,B)
% %
% %    A - (DxM) matrix 
% %    B - (DxN) matrix
% %    df = 1, force diagonals to be zero; 0 (default), do not force
% % 
% % Returns:
% %    E - (MxN) Euclidean distances between vectors in A and B
% %
% %
% % Description : 
% %    This fully vectorized (VERY FAST!) m-file computes the 
% %    Euclidean distance between two vectors by:
% %
% %                 ||A-B|| = sqrt ( ||A||^2 + ||B||^2 - 2*A.B )
% %
% % Example : 
% %    A = rand(400,100); B = rand(400,200);
% %    d = L2_distance(A,B);

% % Author   : Roland Bunschoten
% %            University of Amsterdam
% %            Intelligent Autonomous Systems (IAS) group
% %            Kruislaan 403  1098 SJ Amsterdam
% %            tel.(+31)20-5257524
% %            bunschot@wins.uva.nl
% % Last Rev : Wed Oct 20 08:58:08 MET DST 1999
% % Tested   : PC Matlab v5.2 and Solaris Matlab v5.3

% % Copyright notice: You are free to modify, extend and distribute 
% %    this code granted that the author of the original code is 
% %    mentioned as the original author of the code.

% % Fixed by JBT (3/18/00) to work for 1-dimensional vectors
% % and to warn for imaginary numbers.  Also ensures that 
% % output is all real, and allows the option of forcing diagonals to
% % be zero.  

% if (nargin < 2)
%    error('Not enough input arguments');
% end

% if (nargin < 3)
%    df = 0;    % by default, do not force 0 on the diagonal
% end

% if (size(a,1) ~= size(b,1))
%    error('A and B should be of same dimensionality');
% end

% if ~(isreal(a)*isreal(b))
%    disp('Warning: running distance.m with imaginary numbers.  Results may be off.'); 
% end

% if (size(a,1) == 1)
%   a = [a; zeros(1,size(a,2))]; 
%   b = [b; zeros(1,size(b,2))]; 
% end

% aa=sum(a.*a); bb=sum(b.*b); ab=a'*b;   %求得两两列向量内积，每个列向量模平方
% d = sqrt(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

% % make sure result is all real
% d = real(d); 

% % force 0 on the diagonal? 
% if (df==1)
%   d = d.*(1-eye(size(d)));    %强制所有对角元素为0,为什么要强制？不存在自己和自己的距离 向量来自不同矩阵
% end


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
