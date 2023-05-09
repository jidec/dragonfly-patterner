
# wing_data=wings
# pattern_dir="D:/wing-color/data/patterns/grouped/size_normalized"
# filter_ids=NA
# wing_type="hind"
# group_by_species=TRUE
# only_males_with_females = FALSE
# target_channel1_1=0.4549
# target_channel1_2=0.7372
# target_channel1_3=0.1843
# target_channel2_1=0.4549
# target_channel2_2=0.7372
# target_channel2_3=0.1843
# target_channel3_1=0.4549
# target_channel3_2=0.7372
# target_channel3_3=0.1843
# xPC=1
# yPC=2
# title="Predicted"
# color_by_flight_style=FALSE
# color_percent_cutoff=0.05

# return a pca of pattern element(s) specified through one or more target channels
makePatternPCA <- function(wing_data,pattern_dir="D:/wing-color/data/patterns/grouped/size_normalized",filter_ids=NA,wing_type="fore",
                           group_by_species=TRUE, only_males_with_females = FALSE, color_by_flight_style=FALSE,
                           target_channel1_1=0.4549,target_channel1_2=0.7372,target_channel1_3=0.1843,
                           target_channel2_1=0.4549,target_channel2_2=0.7372,target_channel2_3=0.1843,
                           target_channel3_1=0.4549,target_channel3_2=0.7372,target_channel3_3=0.1843,
                           color_percent_cutoff=NULL,
                           xPC=1,yPC=2,title="Predicted"){
  library(png)
  library(RNiftyReg)
  library(patternize)
  library(stringr)

  print("Loading patterns...")
  # load all patterns
  imgs <- list()
  ids <- c()
  i <- 0
  for(f in list.files(pattern_dir,full.names = TRUE)){
    id <- str_split_fixed(str_split_fixed(f,"/",8)[[7]],"_",3)[[1]]
    if((is.na(filter_ids) | id %in% filter_ids) & str_detect(f,wing_type)){
      img <- readPNG(f)
      if(!is.null(color_percent_cutoff)){
          arr <- as.array(img)
          og_dim <- dim(arr)
          dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4)
          #clust_mask <- abs(arr[,1] - target_channel1_1) < 0.0001 | abs(arr[,1] - target_channel1_2) < 0.0001 | abs(arr[,1] - target_channel1_3) < 0.0001
          clust_mask <- abs(arr[,1] - target_channel1_1) < 0.0001 | abs(arr[,1] - target_channel1_2) < 0.0001
          if(sum(clust_mask)/(og_dim[1] * og_dim[2]) > color_percent_cutoff){
              imgs <- append(imgs,list(img))
              ids <- c(ids, id)
          }
      }
      else{
          imgs <- append(imgs,list(img))
          ids <- c(ids, id)
      }
    }
    i <- i + 1
    #print(i)
  }

  # load reference
  #ref <- readPNG(ref_path)

  # align all patterns to reference
  #i = 1
  #print("Aligning patterns to reference using rNiftyReg...")
  #for(i in 1:length(imgs)){
  #  imgs[[i]] <- niftyreg(imgs[[i]], ref)$image
  #  if(i%%10 == 0){
  #    print(i)
  #  }
  #}

  cluster_masks <- list()

  print("Getting raster masks for cluster")
  # get raster masks for cluster
  for(i in 1:length(imgs)){

    # convert to arr
    arr <- as.array(imgs[[i]])
    # save original dims
    og_dim <- dim(arr)

    # reshape to pixels
    dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4)
    # get 1/NA mask of where cluster occurs
    #clust_mask <- abs(arr[,1] - target_channel1_1) < 0.0001 | abs(arr[,1] - target_channel1_2) < 0.0001 | abs(arr[,1] - target_channel1_3) < 0.0001
    clust_mask <- abs(arr[,1] - target_channel1_1) < 0.0001 | abs(arr[,1] - target_channel1_2) < 0.0001
    clust_mask[clust_mask == TRUE] <- 1
    clust_mask[clust_mask == FALSE] <- NA

    # reshape back to 2d array
    dim(clust_mask) <- c(og_dim[1],og_dim[2])

    # convert to RasterLayer and add to list
    library(raster)
    clust_mask <- raster(clust_mask)
    cluster_masks <- append(cluster_masks,list(clust_mask))
    #print(i)
  }

  # match ids of patterns to wing_data
  matched <- cbind(ids,wing_data[match(ids,wing_data$uniq_id),])

  if(group_by_species){
    sp <- matched$species
    ids_for_sp_indv <- c()
    uniq_sp <- unique(sp)

    cluster_masks2 <- list()

    # for each unique species
    for(s in uniq_sp){
      species_has_m_and_f <- nrow(dplyr::filter(matched,sp==s & Sex=="M")) > 0 & nrow(dplyr::filter(matched,sp==s & Sex=="F")) > 0
      if(species_has_m_and_f | !only_males_with_females){
        for(sex in c("M","F")){
          id <- dplyr::filter(matched,sp==s & Sex==sex)[1,1]
          #print(id)
          pos_in_patterns <- match(id,matched$ids)
          ids_for_sp_indv <- c(ids_for_sp_indv,id)
          cluster_masks2 <- append(cluster_masks2,cluster_masks[[pos_in_patterns]])
        }
      }
    }
    cluster_masks <- cluster_masks2
    ids <- ids_for_sp_indv
    # rematch
    matched <- cbind(ids,wing_data[match(ids,wing_data$uniq_id),])
  }

  popList <- list('None')
  colList <- c("blue","red")
  symbolList <- c(15,16)

  sex_index <- match("Sex", colnames(matched))
  f <- matched[matched[,sex_index] == "F",][,1] # make sure this col # for Sex doesnt change!!
  m <- matched[matched[,sex_index] == "M",][,1]

  if(color_by_flight_style){
    flight_index <- match("flight_type_rm_inter",colnames(matched))
    f <- matched[matched[,flight_index] == "flyer",][,1]
    m <- matched[matched[,flight_index] == "percher",][,1]
  }


  popList <- list(m, f)

  pcaOut <- patPCA(cluster_masks, popList, colList,
                   plot = TRUE, PCx=xPC,PCy=yPC,plotType = 'points', plotChanges = TRUE,
                   plotCartoon = FALSE, refShape = 'target',normalized=TRUE,legendTitle =title)#, refImage = cluster_masks[[1]])#, refShape = 'target', cartoonID = '1',outline = outline_BC0057)
  print(summary(pcaOut[[3]]))
  print(pcaOut[[3]]$sdev)
  # names(cluster_masks) <- c('1','2','3','4','5')
  # look to see if PCs are heteroskedastic
  # symbolList

  return(pcaOut)
}
