library(dplyr)

data <- read.csv('~/dev/others/personal/tag_walk/data/assocs.csv')

data.aug <-
  data %>%
  dplyr::group_by(image) %>%
  dplyr::summarise(
    tag_count = n(),
    agg = paste(tag, collapse = ",")
  )
