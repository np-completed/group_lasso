# Given an h2o learning function and realted parameters,
# evaluate its performance on the sonar data set

h2o.experiment = function(h2o.learner, parameters) {
  base.parameters = list(x = 1:60,
                         y = 61,
                         training_frame = sonar.train)
  final.parameters = if (missing(parameters))
    base.parameters
  else
    c(base.parameters, parameters)
  fit = do.call(h2o.learner, final.parameters)
  
  predictions = h2o.predict(fit, sonar.test)
  p =  h2o.performance(fit, sonar.test)
  plot(p, pch=20, col="blue", cex=0.8)
  lines(lowess(h2o.fpr(p)$fpr, h2o.tpr(p)$tpr, f=1/10), col="blue", lwd=2)
  show(p)
  p
}


