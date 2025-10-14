#### These guidelines will instruct you on how to easily calculate inter-annotator agreement!
As complimentary material, I will provide some toy data that can show you what the input should look like.
There is a nice package in R that will let you calculate a wide variety of agreement metrics with only a few lines of code. No hassle, and you have be certain of the results.
To run the scripts, I would install R (https://cran.rstudio.com/) and R studio (like Pycharm): https://posit.co/download/rstudio-desktop/ 

With this introduction to agreement studies, I will introduce three settings:

1. if you have two annotators, all samples should be labeled by both annotators. You can then make use of the most commonly used agreement measure: Cohen's kappa.
2. if you have than two annotators and each annotator has labeled all samples, you should use Fleiss' kappa.
3. if you have more than two annotators, but they did not all annotate the full dataset (only a subset of the samples), you can still calculate agreement, as long as each sample is labeled more than once. For this purpose, you will need Krippendorff's alpha.


## For starters, We assume a scenario where you have labeled samples for a binary classification task, and you have two annotators.
Your input file (binary_kappa.csv) should have a row for each labeled sample and at least two columns that include the labels (in binary values), one for each annotator.

1. You will first need to read in this file with R. To read in a file and set the content to a variable, you can use this line of code:
labeled_data <- read.csv("toy_data/binary_kappa.csv")

2. You can then print the table as such:
labeled_data

3. To calculate Cohen's Kappa, you need to create a contingency table based on the two columns
contingency_table <- table(labeled_data$Rater_0, labeled_data$Rater_1)

4. then, we can calculate the kappa scores by calling the package for inter-rater agreement (irrCAC):
irrCAC::kappa2.table(contingency_table)


## For the second use case, where we have more than two annotators, we use Fleiss' kappa.
Fleiss kappa also works on the same input file as for Cohen's kappa, but supports more than 2 annotators. Make sure each annotator labelled the full dataset and not just a subset of the data!
To calculate Fleiss' kappa

1. load in the data again
labeled_data <- read.csv("toy_data/binary_kappa.csv")

2. immediately call the package to calculate agreement on the annotated data:
irrCAC::fleiss.kappa.raw(labeled_data)


## For the third use case, you included many different annotators and each individual only annotated a part of the complete dataset. For this you need to use Krippendorff's alpha.
As shown in the binary_alpha.csv file, you again need to make sure that each annotator has their own column and that each sample has its own row. If an annotator did not label a sample, the value should be empty.

1. load in the data
labeled_data <- read.csv("toy_data/binary_alpha.csv")

2. simply run the command for Krippendorff's alpha, this will take care of everything for you:
irrCAC::fleiss.kappa.raw(labeled_data)


### What if you did not label for binary classification?
If you have more than two labels, these scripts still work!

1. Multi-class classification
However, you should keep in mind that the default weighting is set to "binary" this means that the measure considers an equal labels distance. So if you are annotating for emotion detection, you may have 5 categorical labels and a mismatch between "fear" + "joy" will be considered the same as a mismatch between "love" and "joy". This is not a problem but just keep this in mind.

2. Ordered labels
If you used a Likert scale instead of binary classification, labeling with ordered values from 1-7 or 1-5, this binary weighting may not be ideal for calculating agreement. Because the label distance between "1" and "7" should be larger than for "1" and "3".
To address this, you can change the weighting strategy from "binary" to "linear". You can do this for each of the agreement metrics as such:

irrCAC::kappa2.table(contingency_table, weights = "linear")
irrCAC::fleiss.kappa.raw(labeled_data, weights = "linear")
irrCAC::krippen.alpha.raw(labeled_data, weights = "linear")