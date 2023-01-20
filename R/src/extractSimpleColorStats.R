# path = "../experiments/odo_seg_analyzer/images/4155610_198_discrete.png"
#img_dir = "E:/dragonfly-patterner/data/patterns/gomph_grouped_5000"
#img_dir = "D:/wing-color/data/patterns/grouped2"
#start_index = 1
extractSimpleColorStats <- function(img_dir, start_index=1){
    library(imager)
    library(sqldf)
    library(dplyr)
    
    # get paths
    paths <- list.files(img_dir, full.names = TRUE)
    paths <- paths[start_index:length(paths)]
    
    # loop through first to find all unique colors
    all_colors <- c()
    i <- 0
    for(path in paths){
      if(i %% 100 == 0){
        print(i)
      }
      i <- i + 1
      if(file.exists(path) & endsWith(path,".png")){
      
        img <- load.image(path) # load image 
        arr <- as.array(img) # convert to arr
        dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4) # reshape to pixels
        
        # remove black background pixels 
        arr <- arr[(arr != c(0,0,0,0))[,1],]
        
        # get name (id) from path
        name <- strsplit(path, "/", fixed = TRUE)[[1]]
        name <- name[length(name)]
        name <- strsplit(name, "_", fixed = TRUE)[[1]]
        name <- name[1]
        
        # get unique colors
        colors <- unique(arr)
        colors <- colors[order(colors[,1],decreasing=FALSE),]
        
        all_colors <- rbind(all_colors,colors)
      }
    }

    uniq_colors <- all_colors[!duplicated(all_colors),]
    
    # loop through again to get props and build df
    df <- data.frame()
    i <- 0 
    for(path in paths){
        if(i %% 100 == 0){
            print(i)
        }
        i <- i + 1
        
        if(file.exists(path) & endsWith(path,".png")){
            img <- load.image(path)
            arr <- as.array(img)

            # reshape to pixels
            dim(arr) <- c(dim(arr)[1] * dim(arr)[2],4)
            
            # remove black background pixels 
            arr <- arr[(arr != c(0,0,0,0))[,1],]
            
            name <- strsplit(path, "/", fixed = TRUE)[[1]]
            name <- name[length(name)]
            name <- strsplit(name, "_", fixed = TRUE)[[1]]
            name <- name[1]

            # create row for image
            row <- c(name)
            colnames <- c("recordID")
            # add mean color to row
            mean_rgb <- c(mean(arr[,1]), mean(arr[,2]), mean(arr[,3]))
            row <- c(row,mean_rgb)
       
            colnames <- c(colnames,"mean_r", "mean_g","mean_b")
    
            # for every color
            k <- 1
            c <- 1
            if(!is.null(nrow(uniq_colors))){
            for(c in 1:nrow(uniq_colors)){
                color <- uniq_colors[c,]
  
                col_pix <- sum(arr[,1] ==  color[1] & arr[,2] == color[2] & arr[,3] == color[3])
                total_pix <- nrow(arr)
  
                # calculate proportion of pixels
                prop <- col_pix/total_pix
    
                # add the color and prop color to row 
                row <- c(row,color[1:3],prop)

                #print(k)
                #colnames <- c(colnames,paste0("col_",as.character(k),"_r"),
                #              paste0("col_",as.character(k),"_g"),
                #              paste0("col_",as.character(k),"_b"),
                #              paste0("col_",as.character(k),"_prop"))
        
                k <- k + 1
            }
            }
        colnames(df) <- NULL
        colnames(row) <- NULL
        df <- rbind(df,row)
        }
    }
    
    colnames <- c("recordID")
    colnames <- c(colnames,"mean_r", "mean_g","mean_b")
    
    for(c in 1:nrow(uniq_colors)){
      colnames <- c(colnames,paste0("col_",as.character(c),"_r"),
                    paste0("col_",as.character(c),"_g"),
                    paste0("col_",as.character(c),"_b"),
                    paste0("col_",as.character(c),"_prop"))
    }
    
    colnames(df) <- colnames
    df[,2:ncol(df)] <- as.data.frame(sapply(df[,2:ncol(df)], as.numeric))
    df$mean_lightness <- rowMeans(df[,3:5])
    
    return(df)
}

#wing_colors <- extractSimpleColorStats("D:/wing-color/data/patterns")
#img_dir <- "D:/wing-color/data/patterns"
#start_index <- 1000
#View(df)
#as.character(1)
#View(wing_colors)

#install.packages("hablar")
