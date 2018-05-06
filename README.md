# group lasso

The `group_lasso` project performs regularization using the Sonar dataset.


# Background
This README file shows the technical implementation for statistical regularization. All discussion, results and background are on display as PDF or HTML docs, associated with a `.Rmd` file from `R`.

- [ ] Setup the WIKI page of GitHub for this project.

- [ ] Remove missing data command `Sonar_completeCase = Sonar[complete.cases(Sonar),]` , bc there is no missing data.

- [ ] Migrate to `sparklyr` for utilizing `H2O`.

- [ ] Group YML file for parent-child docs


# GitHub Setup:

Create new repository from command line

```
echo "# group_lasso" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/prfrl/group_lasso.git
git push -u origin master
```

## Project Directories

THe GitHub directories follow my local project directories. See
 `directory_structure.png` to see a picture.

The directory tree is logically formatted, using best practices (maybe).
All files are placed relative to the root, where the `.gitignore` file
is located.


* code
  * deep_learn_h2o_roxy.R **Deep Learning** Machine learning models run in
   `H2O`.File uses roxygen formatting.
  * deep_learn_h2o_plain.R  A non-roxygen version kept as backup, just in case.
  * helper_h2o.R
  * grp_lasso.Rmd    **Main File** Parent (main) file aggregates child R files.    
  * grp_lasso.html
  * grp_lasso_cache/html  Cache files are ignored to save space on GitHub.
  * grp_lasso_files/figure-html Many plots are ignored, to save space.

# requirements

## R libraries
The knitr md file `group_lasso` generated the HTML output using R libraries:
`group_lasso/code/grp_lasso_cache/html/__packages` . All bleeding edge packages also have install directions shown in the `.Rmd` file. Typically `library(devtools)` and `install_github(foo_bar/magick)` .


# Roxygen formatting

R files formatted using roxygen can be rendered using pandoc, command line
 arguments using a bash shell OR R. Most markdown effects persist, however
 the main two changes are:

1.  Plain texts in roxygen comments `#'` are preserved as normal texts
  (may contain inline R code)

1. chunk options are written after `#+` or `#-`, e.g.
`#+ chunk-label, opt1=value1, opt2=value2`

There are three main ways to run roxygen: https://cran.r-project.org/web/packages/roxygen2/vignettes/roxygen2.html

1. `roxygen2::roxygenise()`
1. `devtools::document()` , if you’re using devtools
1. `Ctrl + Shift + D`, if you’re using RStudio.

Ideally a main YML file is used and pulled from all existing parent child files.
This has not happened yet.

## Parent File Output from Multiple Child Files

Simple solution is to run `source(foo_bar.R)` from the parent file. this
 is simple but not flexible. Output of tables and plots can be formatted
 using HTML from the plot figures. Variables are callable without too much
 fuss.

Compiling repors from R scripts:
 https://rmarkdown.rstudio.com/articles_report_from_r_script.html
