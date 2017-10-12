rmarkdown_name <- "HW5"
system(sprintf("pandoc %s.md -s -o %s.html -S", rmarkdown_name, rmarkdown_name))
