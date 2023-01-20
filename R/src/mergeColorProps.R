
mergeColorProps <- function(color_stats_df, color1_prop_index, color2_prop_index){
  out_df <- color_stats_df
  out_df[,color1_prop_index] <- out_df[,color1_prop_index] + out_df[,color2_prop_index]
  return(out_df)
}