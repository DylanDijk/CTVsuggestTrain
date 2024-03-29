
<!-- README.md is generated from README.Rmd. Please edit that file -->

# CTVsuggestTrain

<!-- badges: start -->
<!-- badges: end -->

CTVsuggestTrain carries out the model building, and creates the
`data.frame` containing the classification probabilities that is
outputted by the
[CTVsuggest](https://dylandijk.github.io/CTVsuggest/index.html) package.  
These R packages are based on follow up work of my 4<sup>th</sup> year university [dissertation](https://dylandijk.github.io/assets/pdf/Dissertation.pdf) supervised by [Ioannis Kosmidis](https://www.ikosmidis.com/)

The CTVsuggestTrain R package has a single exported function:
`Train_model()`, that constructs features and trains a multinomial
logistic regression model with the objective of classifying CRAN
packages to available [CRAN Task
Views](https://github.com/cran-task-views/ctv#available-task-views). For
a more detailed description of the model, view the [Model
Section](https://dylandijk.github.io/CTVsuggest/articles/CTVsuggest-Overview.html#the-model)
of the [CTVsuggest Overview
Vignette](https://dylandijk.github.io/CTVsuggest/articles/CTVsuggest-Overview.html).

Important to note that in order to output suggestions using the
[CTVsuggest](https://dylandijk.github.io/CTVsuggest/index.html) package,
you can completely ignore the CTVsuggestTrain package. I use
CTVsuggestTrain to train the model weekly in order to update the
predictions provided by
[CTVsuggest](https://dylandijk.github.io/CTVsuggest/index.html). Having
the code packaged makes it easier for me to carry out model training,
and allows the model building to be transparent for others to inspect.

For further detail on the workflow, view the [Packages Workflow
Section](https://dylandijk.github.io/CTVsuggest/articles/CTVsuggest-Overview.html#the-package-workflow)
of the [CTVsuggest Overview
Vignette](https://dylandijk.github.io/CTVsuggest/articles/CTVsuggest-Overview.html).

## Installation

You can install the development version of CTVsuggestTrain from GitHub
with:

``` r
# install.packages("devtools")
devtools::install_github("DylanDijk/CTVsuggestTrain")
```

## Example

The following code saves the model, model accuracy and `data.frame`
containing classification probabilities for packages to an `"OUTPUT"`
directory in your current working directory.

``` r
library(CTVsuggestTrain)
Train_model(save_output = TRUE, save_path = "OUTPUT/")
```

The code example above is the code I run to retrieve an up to date
model. The `Train_model()` function takes a while to run, on my machine
(*Windows Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz, 2112 Mhz, 4 Cores,
8 Logical Processors*) it takes 30 minutes.
