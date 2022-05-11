#records <- read.csv("oc_photos.sh",quote = "",sep=" ",header=FALSE)

# parse dragonfly script
downloadOCImages <- function(start_image=1, root_path = "../..")
{
  library(curl)
  library(stringr)
  records <- read.csv("data/oc_curl_records.csv",header=TRUE,row.names = 1)
  records$url <- str_replace_all(records$url,"\"","")
  for(i in start_image:nrow(records))
  {
    print(i)
    url <- records[i,]$url
    print(url)
    img_name <- records[i,]$image_name
    print(img_name)
    curl_download(url = url, destfile = paste0(root_path, "/data/all_images/", img_name))
  }
}

#downloadOCImages(root_path="..")

# NOTE - OC images will contain _oc at the end, iNat images no extension 
# random images should also have 