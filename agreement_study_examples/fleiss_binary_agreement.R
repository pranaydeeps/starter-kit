labeled_data <- read.csv("toy_data/binary_kappa.csv")

irrCAC::fleiss.kappa.raw(labeled_data)
