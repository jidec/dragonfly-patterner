# count number of a specific genus
data <- read.csv("../data/inat_odonata_usa.csv",header=TRUE,row.names=NULL,sep=",")
library(dplyr)

devtools::install_github("jaredhuling/jcolors")

genus_obs <- dplyr::filter(data, genus == "Stylurus")
unique(genus_obs$species) # species in genus 

annotations <- read.csv("../data/annotations.csv",header=TRUE,row.names=NULL,sep=",")
length(annotations$is_perfect == TRUE)

sum(annotations$is_perfect,na.rm=TRUE)
table(annotations$dorsal_lateral_bad)

install.packages("ggplot2")
library(ggplot2)
ggplot(data, aes(x=species)) + 
  geom_histogram(stat="count")

install.packages("forcats")
library(forcats)
library(scales)
ggplot(data,aes(x = fct_infreq(species),fill=fct_infreq(species))) +
  geom_bar(stat = 'count') + scale_y_continuous(breaks=pretty_breaks(20)) + scale_color_gradient()

j <- 7
df <- data.frame(matrix(nrow=j,ncol=2))
df$count_threshold <- numeric(j)
df$num_species <- numeric(j)

t <- c(5,25,100,200,500,1000,5000)
for(i in 1:j){
  ti <- t[i]
  df$num_species[i] <- sum(table(data$species) > ti)
  df$count_threshold[i] <- ti
}

df$count_threshold <- as.factor(df$count_threshold)
df$percent_species <- df$num_species / 450

library(jcolors)
ggplot(df,aes(x=count_threshold,y=percent_species,fill=percent_species)) + geom_bar(stat="identity") + 
  labs(x="Count Threshold",y="Percent of USA Species") + scale_y_continuous(breaks = scales::pretty_breaks(n = 5))+
  geom_text(aes(label=num_species), position=position_dodge(width=0.9), vjust=-0.25) +
  scale_color_gradient()

