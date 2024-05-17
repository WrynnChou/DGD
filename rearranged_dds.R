library(RANN)
library(UniDOE)
library(SPlit)
library(numbers)


GLP<-function(n,p,type="CD2"){
  fb<-c(3,5,8,13,21,34,55,89,144,233,377,610,987)
  if(((n+1)%in%fb)&(p==2)){
    design0<-matrix(0,(n+1),p)
    H<-rep(1,2)
    H[2]<-fb[which(fb==(n+1))-1]
    for (j in 1:p) {
      for (i in 1:(n+1)) {
        design0[i,j]<-(2*i*H[j]-1)/(2*(n+1))-floor((2*i*H[j]-1)/(2*(n+1)))
      }
    }
    design0<-design0[-(n+1),]*(n+1)/n
  }else{
    if(p==1){
      design0<-matrix(0,n,p)
      for(i in 1:n){
        design0[i,1]<-(2*i-1)/(2*n)
      }
      return(design0)
    }
    h<-c()
    for(i in 2:min((n+1),200)){
      if(coprime((n+1),i)==T){
        h<-c(h,i)
      }
    }
    if(p>2){
      for (i in 1:100) {
        if(choose(p+i,i)>5000){
          addnumber<-i
          break
        }
      }
      h<-h[sample(1:length(h),min(length(h),(p+addnumber)))]
    }
    H<-combn(h,p,simplify = F)
    if(length(H)>3000){
      H<-H[sample(3000)]
    }
    design0<-matrix(0,n,p)
    d0<-DesignEval(design0,crit=type)
    for (t in 1:length(H)) {
      design<-matrix(0,n,p)
      for (i in 1:p) {
        for (j in 1:n) {
          design[j,i]<-(j*H[[t]][i])%%(n+1)
        }
      }
      d1<-DesignEval(design,crit=type)
      if(d1<d0){
        d0<-d1
        design0<-design
      }
    }
    design0<-(design0*2-1)/(2*n)
  }
  return(design0)
}


BRUD<-function(Design){
  D=Design
  n=nrow(D)
  s=ncol(D)
  rand=matrix(runif((n*s),0,1),nrow=n,ncol=s)
  eta_mat=matrix(rep(sample(c(0:(n-1)),s,replace = TRUE),n),nrow=n,ncol=s,byrow=TRUE)
  eta_mat=((eta_mat-0.5)/n)
  RUD=(D+eta_mat)%%1+rand/n
  #RUD=1-abs(2*RUD-1)
  return(RUD)
}


DDS_dnn_givendim<-function(data,n,reduced.dim=15,Design=NULL,type="CD2",ratio=0.85,scale=FALSE){
  if((type %in% c("CD2","WD2","MD2"))==F){
    stop("type shoud be chose from 'CD2','WD2','MD2'")
  }
  if(n>dim(data)[1]){
    stop("The subsample size n should be less than sample size N")
  }
  if(scale==TRUE){
    x<-apply(as.matrix(data),2,scale_modi)
  }
  if(scale==FALSE){
    x<-scale(as.matrix(data),scale=FALSE)
  }
  l<-reduced.dim
  y<-n
  N<-dim(x)[1]
  p0<-dim(x)[2]
  svddata<-svd(x)
  print(paste("p choosen =",l))
  rdata<-x%*%svddata$v[,1:l]
  if(is.null(Design)){
    design<-GLP(n,l,type)
  }
  else{
    design<-BRUD(Design)
  }
  yita<-matrix(nrow=N,ncol=l)
  for (i in 1:l) {
    fn=ecdf(rdata[,i])
    yita[,i]=fn(rdata[,i])
  }
  kdtree<-nn2(data=yita,query=design,k=1,treetype="kd")
  subsample<-kdtree$nn.idx
  return(subsample)
}


seq_DDS_dnn_givendim<-function(data,n,k,reduced.dim=15,Design=NULL,type="CD2",ratio=0.85,scale=FALSE){
  if((type %in% c("CD2","WD2","MD2"))==F){
    stop("type shoud be chose from 'CD2','WD2','MD2'")
  }
  if(n>dim(data)[1]){
    stop("The subsample size n should be less than sample size N")
  }
  td<-reduced.dim
  if(is.null(Design)){
    design=GLP(n,td,type)
  }
  else{
    design=Design
  }
  indices<-array()
  data_collection<-list()
  un_sampled_collection<-list()
  N<-dim(data)[1]
  indices_collection<-CVgroup(k,N,100)
  for(i in 1:k){
    data_collection[[i]]=data[indices_collection[[i]],]
    un_sampled_collection[[i]]=DDS_dnn_givendim(data=data_collection[[i]],n=n,reduced.dim=td,Design=design,type=type,ratio=ratio,scale=scale)[,1]
    sampled_indices=indices_collection[[i]][un_sampled_collection[[i]]]
    indices=c(indices,sampled_indices)
  }
  indices=indices[-1]
  return(indices)
}


rearrange_DDS_regress_without<-function(data,b,divided.number=1,maximum.iteration=(nrow(data)%/%(b*divided.number))*2,remaining.rate=2*(divided.number*b)/nrow(data),repetitive.rate=0.5,Design=NULL,type='CD2',ratio=0.85,scale=FALSE,reduced.dim=15){
  pp=remaining.rate
  re=repetitive.rate*(b*divided.number)
  #responses should be in the first column of "data"
  d_data=data
  det_data=matrix(ncol=ncol(data))
  n=nrow(data)
  if(divided.number==1){
    m=1
    mm=maximum.iteration
    while(m<mm+1){
      if(is.null(reduced.dim)){
        sa=DDS_dnn(d_data[,-1],b,type,ratio,scale)[,1]
      }
      else{
        sa=DDS_dnn_givendim(d_data[,-1],b,reduced.dim,Design,type,ratio,scale)[,1]
      }
      
      det_data=rbind(det_data,d_data[sa,])
      d_data=d_data[-sa,]
      m=m+1
      if(length(unique(sa))<re){
        break
      }
      if(nrow(d_data)<(pp*n)){
        break
      }
    }
    det_data=rbind(det_data,d_data)[-1,]
    iteration_number=m-1
  }
  else{
    m=1
    mm=maximum.iteration
    while(m<mm+1){
      if(is.null(reduced.dim)){
        sa=seq_DDS_dnn(d_data[,-1],b,divided.number,type,ratio,scale)
      }
      else{
        sa=seq_DDS_dnn_givendim(d_data[,-1],b,divided.number,reduced.dim,Design,type,ratio,scale)
      }
      det_data=rbind(det_data,d_data[sa,])
      d_data=d_data[-sa,]
      m=m+1
      if(length(unique(sa))<re){
        break
      }
      if(nrow(d_data)<(pp*n)){
        break
      }
    }
    det_data=rbind(det_data,d_data)[-1,]
    iteration_number=m-1
    
  }
  return(list(iteration_number=iteration_number,rearrange_data=det_data))
}

cifar_re2500=rearrange_DDS_regress_without(cifar10,500,divided.number = 5,Design=GLP_500_15,reduced.dim = 15)