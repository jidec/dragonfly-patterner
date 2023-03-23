library(plotly)
library(ggplot2)

# plot sexes
ggplot(wings,aes(x=Sex)) + geom_histogram(stat="count")

# plot species
ggplot(wings, aes(x=Species)) + geom_bar()
ggplot(wings, aes(x=Genus)) + geom_bar()

ggplot(wings,aes(col_2_prop,col_6_prop,colour = Sex)) + geom_point() + 
  xlab("Percent of wing that is brown") + ylab("Percent of wing that is black")

library(RColorBrewer)
getPalette = colorRampPalette(brewer.pal(9, "Set1"))
# create an interactive plot showing how families have similar amounts of pigments
fig <- plot_ly(x=wings$col_2_prop, y=wings$col_6_prop, z=wings$col_1_prop, 
        type="scatter3d", mode="markers", color=wings$Family,colors=getPalette(50)) 
axx <- list(title = "% wing brown")
axy <- list(title = "% wing black")
axz <- list(title = "% wing yellow")
fig %>% layout(scene = list(xaxis=axx,yaxis=axy,zaxis=axz))

# same with sexes
fig <- plot_ly(x=wings$col_2_prop, y=wings$col_6_prop, z=wings$col_1_prop, 
               type="scatter3d", mode="markers", color=wings$Sex,colors=getPalette(50)) 
axx <- list(title = "% wing brown")
axy <- list(title = "% wing black")
axz <- list(title = "% wing yellow")
fig %>% layout(scene = list(xaxis=axx,yaxis=axy,zaxis=axz))

# same with hind fore
fig <- plot_ly(x=wings$col_2_prop, y=wings$col_6_prop, z=wings$col_1_prop, 
               type="scatter3d", mode="markers", color=wings$wing_type,colors=getPalette(50)) 
axx <- list(title = "% wing brown")
axy <- list(title = "% wing black")
axz <- list(title = "% wing yellow")
fig %>% layout(scene = list(xaxis=axx,yaxis=axy,zaxis=axz))

