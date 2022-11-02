tree <- odonate_tree

library(stringr)
#genus <- "Pinheyschna"
str_split_fixed(tree$tip.label, " ", 2)[,1]
#species<-grep("Pinheyschna",tree$tip.label)
tips <- c()
for(genus in unique(str_split_fixed(tree$tip.label, " ", 2)[,1])) {
    tip <- grep(genus,tree$tip.label)[1]
    tips <- c(tips,tip)
}
tree <- keep.tip(tree,tips)
tree$tip.label <- str_split_fixed(tree$tip.label, " ", 2)[,1]
plot(tree)
length(tree$tip.label)
for(genus in unique(str_split_fixed(tree$tip.label, " ", 2)[,1])) {
    
    species<-grep(genus,tree$tip.label)
    if (length(species)>=2) {
        mrca<-findMRCA(tree,tree$tip.label[species])
        desc <- getDescendants(tree, mrca)
        desc_tips <- desc[desc <= length(tree$tip.label)]
        if (length(grep(genus, tree$tip.label[desc_tips], invert = TRUE))==0) {
            base_node <- mrca - length(tree$tip.label)
            if (is.null(tree$node.label[base_node]) || is.na(tree$node.label[base_node])) {
                tree$node.label[base_node] = genus
            }
        }
    }
}

for(n in 1:length(tree$node.label)){
    node <- tree$node.label[n]
    if(!is.na(node)){
        tree <- bind.tip(tree, node, edge.length=1, where=n, position=0)
    }
}

tree$tip.label
plot(tree)
bind.tip(tree, tip.label, edge.length=1, where=NULL, position=0)
keep.tip(phy, tip)
tree$tip.label
tree$node.label

plot(tree)
unique(sub("_.*", "",tree$tip.label))
sub()