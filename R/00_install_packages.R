# Run once to install the packages used by the R scripts
pkgs <- c("data.table", "ggplot2", "scales", "recipes", "uwot", "jsonlite", "dplyr")
missing <- setdiff(pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing)
invisible(lapply(pkgs, requireNamespace))
cat("All packages available:", paste(pkgs, collapse = ", "), "\n")
