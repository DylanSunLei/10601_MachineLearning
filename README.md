# 10601_MachineLearning

```
Lei Sun;
Xinxin Pan;
```


### Project Part 1 Description 

`Deadline: Wednesday 11/11`

```
Sample Number: 501
Features: 5903
Kind of Labels: {1..5}
Classfication Problem
```

The fMRI scan is taken approximately 6 seconds after the time button would have been pressed, to account for the delay between neural activity and the BOLD signal that the fMRI measures.

Mapping to Label Y

1. Early Stop: Successful stop to an early stop signal.
2. Late Stop: Successful stop to a late stop signal.
3. Correct Go: Correct button press (within ~500ms) on a trial with no stop signal.
4. Incorrect Go: Button press on a trial with stop signal.
5. False Alarm: No button press on a trial with no stop signal.

### Task
Your task for project part 1 is to build the best possible classifier to predict Y from X.
+ Your score will depend on your classifier's accuracy = base credit + additional credit (above threshold)
+ Free to use any tools you like: existing libraries or ones you implement yourself

### Hint
+ You will get decent performance with **a support vector machine using a Gaussian kernel of radius 3×105 and a hinge weight of C=10, using 1-vs-rest and voting**. (baseline)
+ Some issues to think about
+   you can try other classifiers, other methods of reducing multi-class to binary, other kernels, feature engineering, other parameter values, different regularizations, different normalizations
+   make sure to use techniques like **cross-validation, holdout, or bootstrap** within the training data set to avoid fooling yourself about the accuracy of your classifier




