#get list of tip/trait labels that DONT have matching trait/tip data
getMissing <- function(vector2drop, reference)
{
    missing <- vector(mode="character")
    z <- 1
    
    for(i in 1:length(vector2drop))
    {
        tip <- as.character(vector2drop[i])
        match <- FALSE
        for(k in 1:length(reference))
        {
            if(reference[k] == tip)
            {
                match <- TRUE
            }
        }
        if(match == FALSE) {
            missing[z] <- tip
            z <- z + 1
        }
    }
    return(missing)
}

# summarized traits is a df summarized to a clade (often genus or species) with the genus or speices column renamed "clade"
# and a column called "trait
plotPhyloEffects <- function(summarized_traits,ape_tree)
{
    library(ape)
    library(phytools)
    
    # get tips
    tips <- ape_tree$tip.label
    
    # plot tree
    plot(ape_tree, show.tip.label=FALSE)
    
    traits <- summarized_traits
    
    #sort(ape_tree$tip.label)
    #sort(traits$clade)
    #match(traits$clade,ape_tree$tip.label)
    
    # drop tips
    #length(ape_tree$tip.label)
    #length(missing_tips)
    missing_tips <- getMissing(ape_tree$tip.label,traits$clade)
    missing_tips <- as.character(missing_tips)
    ape_tree <- drop.tip(ape_tree,missing_tips)
    
    # drop traits
    missing_traits <- getMissing(traits$clade, ape_tree$tip.label)
    if(length(missing_traits != 0))
    {
        for(i in 1:length(missing_traits))
        {
            traits <- traits[traits[,"clade"] != missing_traits[i],]
        }
    }
    
    traits <- traits[!duplicated(traits$clade),]
    
    # these should now be the same
    print(nrow(traits))
    print(length(ape_tree$tip.label))
    
    
    # prep traits for asr
    trt <- traits$trait #change this to meanMD, meanHour, propNight etc
    
    names(trt) <- traits$clade
    
    # unlist tree tips
    ape_tree$tip.label <- unlist(ape_tree$tip.label)
    
    #makes trait indices match order of tree tips
    matchOrder <- function(tree, traitvect)
    {
        ordered <- traitvect
        for(i in 1:length(tree$tip.label))
        {
            tipstr <- tree$tip.label[i]
            for(ii in 1:length(traitvect))
            {
                if(tipstr == names(traitvect[ii]))
                {
                    ordered[i] <- traitvect[ii]
                    names(ordered)[i] <- tipstr
                }
            }
        }
        return(ordered)
    }
    
    trt <- matchOrder(ape_tree, trt)
    
    # replace NaN edge lengths with 0
    ape_tree$edge.length <-ifelse(is.nan(ape_tree$edge.length),0,ape_tree$edge.length)
    ape_tree$edge.length[ape_tree$edge.length == 0] <- 0.1
    
    # create contMap for all genera
    obj <- contMap(ape_tree,trt,method="anc.ML")
    
    plot(obj,type="fan",legend=0.7*max(nodeHeights(ape_tree)), fsize=c(0.6,0.9),cex=1)
    
    # compute phylogenetic signal
    physig <- phylosig(ape_tree, trt, method="K", test=FALSE, nsim=1000, se=NULL, start=NULL,
                       control=list())
    
    print(physig)
}

removeSpeciesNotInPhylo <- function(data,phylo)
{
    # drop traits
    missing_traits <- getMissing(data$species, phylo$tip.label)
    print(length(missing_traits))
    if(length(missing_traits != 0))
    {
        for(i in 1:length(missing_traits))
        {
            data <- data[data[,"species"] != missing_traits[i],]
        }
    }
    return(data)
}

# return a list of two adjusted data
removePhyloMissing <- function(data,phylo)
{
    # drop traits
    missing_traits <- getMissing(data$species, phylo$tip.label)
    print(length(missing_traits))
    if(length(missing_traits != 0))
    {
        for(i in 1:length(missing_traits))
        {
            data <- data[data[,"species"] != missing_traits[i],]
        }
    }
    
    missing_tips <- getMissing(phylo$tip.label,data$species)
    missing_tips <- as.character(missing_tips)
    phylo <- drop.tip(phylo,missing_tips)
    
    return(list(data,phylo))
}

removeDataPhyloMissing <- function(data,phylo)
{
    library(ape)
    # drop traits
    missing_traits <- getMissing(data$species, phylo$tip.label)
    print(length(missing_traits))
    if(length(missing_traits != 0))
    {
        for(i in 1:length(missing_traits))
        {
            data <- data[data[,"species"] != missing_traits[i],]
        }
    }
    
    missing_tips <- getMissing(phylo$tip.label,data$species)
    missing_tips <- as.character(missing_tips)
    phylo <- drop.tip(phylo,missing_tips)
    
    return(list(data,phylo))
}
