
labeled_data <- read.csv("toy_data/likert_alpha.csv")

labeled_data

irrCAC::krippen.alpha.raw(labeled_data, weights = "linear")