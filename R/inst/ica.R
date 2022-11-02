
library("fastICA")
df <- ants[,3:5]
i <- ica(df, 3, method = c("fast", "imax", "jade"))
# compare mixing matrix recovery
acy(Bmat, i$M)
acy(Bmat, i$M)
acy(Bmat, i$M)
# compare source signal recovery
cor(Amat, i$S)
cor(Amat, i$S)
cor(Amat, i$S)

ica <- fastICA(df,2)
plot(ica$X)
plot(ica$X %*% ica$K)
plot(ica$S)

ica$S[,2]