install.packages("rTensor")
library(rTensor)

### How to retrieve faces_tnsr from figshare
faces_tnsr <- load_orl()
subject <- faces_tnsr[,,21,]
#dummy_faces_tnsr <- rand_tensor(c(92,112,40,10))
#subject <- dummy_faces_tnsr[,,21,]
mpcaD <- mpca(subject, ranks=c(10, 10))
mpcaD$conv
mpcaD$norm_percent
plot(mpcaD$all_resids)
mpcaD