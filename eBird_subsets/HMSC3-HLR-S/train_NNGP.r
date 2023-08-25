#install.packages("devtools")
library(devtools)
#install_github("hmsc-r/HMSC", build_opts = c("--no-resave-data", "--no-manual"))
library(Hmsc)
library(MASS)
set.seed(6)


args = commandArgs(trailingOnly=TRUE)

path = "../DATA/"
exp = args[1] #"birds"
num = args[2] #"1"
print(exp)
print(num)


t_S = read.csv(paste(path, sprintf("St_%s_%s.csv", num, exp), sep = ""), header = FALSE)
t_X = read.csv(paste(path, sprintf("Xt_%s_%s.csv", num, exp), sep = ""), header = FALSE)
t_Y = read.csv(paste(path, sprintf("Yt_%s_%s.csv", num, exp), sep = ""), header = FALSE)

v_S = read.csv(paste(path, sprintf("Sv_%s_%s.csv", num, exp), sep = ""), header = FALSE)
v_X = read.csv(paste(path, sprintf("Xv_%s_%s.csv", num, exp), sep = ""), header = FALSE)
v_Y = read.csv(paste(path, sprintf("Yv_%s_%s.csv", num, exp), sep = ""), header = FALSE)

n = dim(t_S)[1] + dim(v_S)[1]
sprintf("n=%d", n)

S = rbind(t_S, v_S)
X = rbind(t_X, v_X)
Y = rbind(t_Y, v_Y)

S = as.matrix(S, nrow=n)
X = as.matrix(X, nrow=n)
Y = as.matrix(Y, nrow=n)

colnames(S) = c("x-coordinate","y-coordinate")
rownames(S) = 1:n

print(dim(S))
print(dim(X))
print(dim(Y))

colnames(X) = 1:dim(X)[2]

XData = data.frame(x1=X[,])
#print(XData[1,])

###########################################
#Knots = constructKnots(S)

studyDesign = data.frame(sample = as.factor(1:n))
rL.spatial = HmscRandomLevel(sData = S, sMethod = 'NNGP', nNeighbours = 10) # sKnot =Knots) 
rL.spatial = setPriors(rL.spatial, nfMin=1, nfMax=1) #We limit the model to one latent variables for visualization purposes
m.spatial = Hmsc(Y=Y, X=X, #XFormula=~x1,
studyDesign=studyDesign, ranLevels=list("sample"=rL.spatial),distr="probit")


nChains = 1
thin = 80
samples = 100
transient = 2000
verbose = 0


m.spatial = sampleMcmc(m.spatial, thin = thin, samples = samples, transient = transient,
nChains = nChains, verbose = verbose,updater=list(GammaEta=FALSE))
print("MCMC is done")

#preds.spatial = computePredictedValues(m.spatial)
#print(dim(preds.spatial))
#MF.spatial = evaluateModelFit(hM=m.spatial, predY=preds.spatial)
#print(MF.spatial)

#Predictive power
partition = c(rep(1, dim(t_S)[1]), rep(2, dim(v_S)[1])) #createPartition(m.spatial, nfolds = 2, column = "sample")
#print(partition)
cvpreds.spatial = computePredictedValues(m.spatial, partition=partition, updater=list(GammaEta=FALSE))
print(dim(cvpreds.spatial))
cvMF.spatial = evaluateModelFit(hM=m.spatial, predY=cvpreds.spatial)
print(cvMF.spatial)

write.csv(cvpreds.spatial, sprintf("NNGP/res_%s_%s.csv", exp, num))
print("END")



