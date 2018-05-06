# group lasso

# Background
This README file shows the technical implementation for statistical regularization. All discussion, results and background are on display as PDF or HTML docs, associated with a `.Rmd` file from `R`.

[] Setup the WIKI page of GitHub for this project.
[] Remove missing data command `Sonar_completeCase = Sonar[complete.cases(Sonar),]` , bc there is no missing data.
[] Migrate to `sparklyr` for utilizing `H2O`.

The `group_lasso` project performs regularization using the Sonar dataset.


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

## GitHub directories

The directory tree is logically formatted. All files are placed relative to the root, where the `.gitignore` file is located.


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
