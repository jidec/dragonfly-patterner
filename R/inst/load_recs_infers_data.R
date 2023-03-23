# import inferences
inferences <- read.csv("E:/dragonfly-patterner/data/inferences.csv")
  
# import inat gbif records
gbif_records <- read.csv("E:/dragonfly-patterner/data/other/raw_records/inatdragonflyusa.csv",sep='\t')

# import genus records
source("src/loadGenusDLRecords.R")
records <- loadGenusDLRecords(write=TRUE)
# the way to do this is to write to raw records, then it should get merged 

# merge records into inferences 
recs_infers <- merge(records,inferences, by="imageID")

# import training data
training <- read.csv("E:/dragonfly-patterner/data/train_metadata.csv")
library(stringr)
training$recordID <- str_replace(training$imageID,"RANDOM","")