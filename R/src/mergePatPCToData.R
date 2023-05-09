# wings_data = wings
# pca = fore_pca_all
# pc_suffix = "fore"
# pca = hind_pca_all


mergePatPCToData <- function(wings_data,pca,pc_suffix){
  library(dplyr)
  # get matrix of pc values and corresponding id
  pc_mat <- pca[[3]]$x[,1:3]
  ids <- pca[[2]]$sampleID

  pc_mat <- cbind(pc_mat,ids)

  # name cols and merge into wings data
  names <- c(paste0("PC1_",pc_suffix),paste0("PC2_",pc_suffix),paste0("PC3_",pc_suffix))
  colnames(pc_mat) <- c(paste0("PC1_",pc_suffix),paste0("PC2_",pc_suffix),paste0("PC3_",pc_suffix),"recordID")
  wings_data <-  merge(wings_data,pc_mat,by="recordID",all.x=TRUE)

  # make pcs numeric
  wings_data <- wings_data %>%
    mutate_at(vars(names), as.numeric)

  return(wings_data)
}
