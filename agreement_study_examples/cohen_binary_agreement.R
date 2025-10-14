
labeled_data <- read.csv("toy_data/binary_kappa.csv")

labeled_data

contingency_table <- table(labeled_data$Rater_0, labeled_data$Rater_1)

irrCAC::kappa2.table(contingency_table)