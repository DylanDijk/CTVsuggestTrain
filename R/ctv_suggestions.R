# Objects I need to be saved:
  # The `predicted_probs_for_suggestions` object can then just filter this.
  # Need to have script that I can schedule that saves the `predicted_probs_for_suggestions` object

  # An idea is to have a `train_model()` function that allows the user to train the model
  # themselves so they have an up to date model instead of relying on my scheduling.

library(pins)
library(glmnet)


board = board_folder(path = "C:/Users/Dylan Dijk/Documents/Projects/CRAN-Task-Views-Recommendations/Pins_board/")


ctv_suggestions = function(taskview = "Econometrics", n = 5){

  predicted_probs_for_suggestions = board %>% pin_read("predicted_probs_for_suggestions")

  suggestions = predicted_probs_for_suggestions[,c(paste0(taskview), "Packages"), drop = F][order(predicted_probs_for_suggestions[,paste0(taskview), drop = F], decreasing = T),, drop = F][1:n,]
  return(suggestions)
}
