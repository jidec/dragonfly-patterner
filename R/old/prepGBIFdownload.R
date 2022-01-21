library("dplyr")
library("sqldf")
install.packages("data-table")
library(data.table)
# Download GBIF archive of research grade observations
# Load as a dataframe

data <- read.table("0110161-210914110416597/0110161-210914110416597.csv",
                   header = TRUE, sep = "\t", fill = TRUE)

View(data)
f <- file("0110161-210914110416597/0110161-210914110416597.csv")
data <- sqldf("select * from f", dbname = tempfile(), file.format = list(header = T, row.names = F))

# Filter for Odonata in USA

data <- filter(data,taxon_order_name = "Odonata")
data <- filter(data,place_country_name = "United States")

write.csv(data,"inat_usa_odonata")

# Get all Odonata genera in USA

genera <- unique(data$taxon_genus_name)
