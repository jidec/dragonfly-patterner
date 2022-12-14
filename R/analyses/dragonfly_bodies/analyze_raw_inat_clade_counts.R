# analyze and plot distributions of species 
inat_records <- read.csv("../data/other/raw_records/inatdragonflyusa.csv",header=TRUE,row.names=NULL,sep="\t")

library(ggplot2)
library(scales)
# plot species counts
ggplot(inat_records, aes(x=species)) + 
    geom_histogram(stat="count")

# plot species counts
ggplot(inat_records, aes(x=genus)) + 
    geom_histogram(stat="count")

ggplot(inat_records, aes(x=family)) + 
    geom_histogram(stat="count") +
    scale_y_continuous(name="count", labels = scales::comma) +
    stat_bin(
        aes(x = family,
            y = after_stat(count),
            label = after_stat(ifelse(count == 0, "", count))),
        binwidth = 4, geom = "text", vjust = -1
    )



length(unique(data$genus))

getThresholdCounts <- function(clade_col,threshes,total){
    # reformat
    t <- threshes
    sp_counts <- data.frame(matrix(nrow=length(t),ncol=2))
    j <- length(threshes)
    sp_counts$count_threshold <- numeric(j)
    sp_counts$num_species <- numeric(j)

    for(i in 1:length(t)){
        ti <- t[i]
        sp_counts$num_species[i] <- sum(table(clade_col) > ti)
        sp_counts$count_threshold[i] <- ti
    }
    
    sp_counts$count_threshold <- as.factor(sp_counts$count_threshold)
    sp_counts$percent_species <- sp_counts$num_species / total
    return(sp_counts)
}

sp_counts <- getThresholdCounts(inat_records$species,c(5,25,100,200,500,1000,5000),total=450)
genus_counts <- getThresholdCounts(inat_records$genus,c(25,100,200,500,1000,5000,10000,20000),total=94)



# plot
ggplot(sp_counts,aes(x=count_threshold,y=percent_species)) + geom_bar(stat="identity") + 
    labs(x="Count Threshold",y="Percent of USA Species") + scale_y_continuous(labels=scales::percent, breaks = scales::pretty_breaks(n = 5))+
    geom_text(aes(label=num_species), position=position_dodge(width=0.9), vjust=-0.25) + 
    ggtitle("USA Odonate Species in iNaturalist")
    #annotate(geom = 'text', label = 'Number of ', x = -Inf, y = Inf, hjust = -5, vjust = 1)
#+ scale_color_gradient()
#+ 
# plot
ggplot(genus_counts,aes(x=count_threshold,y=percent_species)) + geom_bar(stat="identity") + 
    labs(x="Count Threshold",y="Percent of USA Genera") + scale_y_continuous(labels=scales::percent, breaks = scales::pretty_breaks(n = 5))+
    geom_text(aes(label=num_species), position=position_dodge(width=0.9), vjust=-0.25) + 
    ggtitle("Number of USA Odonate Genera in iNaturalist Exceeding Count Thresholds")