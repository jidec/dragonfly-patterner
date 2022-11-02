# path = "../experiments/odo_seg_analyzer/images/4155610_198_discrete.png"
img_dir = "E:/dragonfly-patterner/data/patterns/gomph_grouped_5000"
start_index = 1
extractSimpleColorStats <- function(img_dir, start_index=1){
    library(imager)
    df <- data.frame()
    
    #names <- list.files(img_dir, full.names = FALSE)
    #names <- do.call(rbind,strsplit(names, "_", fixed = TRUE))[,1]
    #names <- names[,1]
    
    paths <- list.files(img_dir, full.names = TRUE)
    paths <- paths[start_index:length(paths)]

    #path <- paths[1]
    i <- 0 
    for(path in paths){
        if(i %% 100 == 0){
            print(i)
        }
        i <- i + 1
        #print(strsplit(names[i], "_", fixed = FALSE)[1][1])
        
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
            # get unique colors
            colors <- unique(arr)
            colors <- colors[order(colors[,1],decreasing=FALSE),]
            #print(colors)
            # for every color
            k <- 1
            
            if(!is.null(nrow(colors))){
              
            
            for(c in 1:nrow(colors)){
                color <- colors[c,]
                
                col_pix <- sum(arr[,1] ==  color[1] & arr[,2] == color[2] & arr[,3] == color[3])
                total_pix <- nrow(arr)
                
                # calculate proportion of pixels
                prop <- col_pix/total_pix
                
                # add the color and prop color to row 
                row <- c(row,color[1:3],prop)
                #print(k)
                colnames <- c(colnames,paste0("col_",as.character(k),"_r"),
                              paste0("col_",as.character(k),"_g"),
                              paste0("col_",as.character(k),"_b"),
                              paste0("col_",as.character(k),"_prop"))
        
                k <- k + 1
            }
            }
        df <- rbind(df,row)
        }
    }
    #print(colnames)
    #print(df[1,])
    colnames(df) <- colnames
    return(df)
}

#wing_colors <- extractSimpleColorStats("D:/wing-color/data/patterns",1)
#img_dir <- "D:/wing-color/data/patterns"
#start_index <- 1000
#View(df)
#as.character(1)
#View(wing_colors)
