
# parse dragonfly script
downloadOCImages <- function(start_image=1, proj_dir = "../..")
{
  library(curl)
  library(stringr)
  records <- read.csv("../data/other/oc_curl_records.csv",header=TRUE,row.names = 1)
  records$query <- str_replace_all(records$query,"\"","")
  for(i in start_image:nrow(records))
  {
    #i = 1
    print(i)
    query <- records[i,]$query
    img_name <- records[i,]$image_name
    img_name <- str_replace(img_name,"-0","-")
    img_name <- paste0("OC-",img_name)
    curl_download(url = query, destfile = paste0(proj_dir, "/data/all_images/", img_name))
  }
}

#downloadOCImages(proj_dir="..")