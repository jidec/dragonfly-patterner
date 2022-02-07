
#install.packages("rinat")
library("rinat")
install.packages("data-table")
# Download images


# for each year
for(y in 2010:2020)
{
    for(m in 1:12)
    {
        data <- rbind(data,get_inat_obs(
            taxon_name = "Odonata",
            place_id = 1,
            quality = 'research',
            geo = NULL,
            year = y,
            month = m,
            maxresults = 10000,
            meta = FALSE
        ))
    }
}
