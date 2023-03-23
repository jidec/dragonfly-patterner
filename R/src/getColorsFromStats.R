getColorsFromStats <- function(data){
  row <- data[1,2:ncol(data)] * 255
  row
  sq <- seq(from=4,ncol(row),by=4)
  to_add <- c()
  
  for(i in 1:length(sq)){
    s <- sq[i]
    to_add <- c(to_add,s+1,s+2)
  }
  sq <- c(sq,to_add)
  sq <- sq[sq < ncol(row)]
  sq <- sq[order(sq)]
  row <- row[sq]
  return(row)
}