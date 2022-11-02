library(phytools)
data(anoletree)
anoletree$maps
mapped.states(anoletree)
cols<-setNames(palette()[1:6],mapped.states(anoletree))
cols
plot(anoletree,cols,type="fan",fsize=0.8,lwd=3,ftype="i")
add.simmap.legend(colors=cols,x=0.9*par()$usr[1],
                  y=0.9*par()$usr[4],prompt=FALSE,fsize=0.9)

eel.tree<-read.tree("data/elopomorph.tre")
eel.data<-read.csv("data/elopomorph.csv",row.names=1)
fmode <- as.factor(setNames(eel.data[,1],rownames(eel.data)))
dotTree(eel.tree,fmode,colors=setNames(c("blue","red"),
                                       c("suction","bite")),ftype="i",fsize=0.7)

eel.tree<-read.tree("data/elopomorph.tre")
eel.data<-read.csv("data/elopomorph.csv",row.names=1)
fmode <- as.factor(setNames(eel.data[,1],rownames(eel.data)))
pa <- as.factor(records$col_1_prop > 0.01)
pa
dotTree(eel.tree,fmode,colors=setNames(c("blue","red"),
                                       c("suction","bite")),ftype="i",fsize=0.7)
