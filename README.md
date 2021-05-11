# Terry Stops #

## Background ##

For a long time now, there has been a rise in tension between law enforcement and citizens. It is believed that alot of it stems from the precedent Supreme Court case of [Terry v. Ohio](https://www.oyez.org/cases/1967/67). This case was a landmark case that found that a police officer was not in violation of the "unreasonable search and seizure" clause of the Fourth Amendment, even though he stopped and frisked a couple of suspects only because their behavior was suspicious. Thus was born the notion of "reasonable suspicion", according to which an agent of the police may e.g. temporarily detain a person, even in the absence of clearer evidence that would be required for full-blown arrests etc. Terry Stops are stops made of suspicious drivers.


Through Seattle's Open Data Program, I was able to download the Terry Stop Data from the past few years, and analyze the data with the purpose of creating a model that would be able to predict whether an arrest was made after a Terry Stop had occurred. 

These are the findings.

## EDA ##

While exploring the data in `Terry_Stops.csv` there were a couple of data cleaning methods to use to make the data useful prior to fitting them to the models.

To make the columns more programming friendly, the columns were renamed to be `snake case`. 

### Feature Creation ###

Using the data found in the csv file, a few features were created that gave more meaningful and specific information that the model could use to predict arrests.

Some of the features include:

- Creating a column that gives the Age of an Officer from the year of birth column (officer_YOB)
    - The age of the officer was calculated from the officer_YOB column and the year that the Terry Stop was reported giving the age of the officer on that given stop.

### Further Data Cleaning ###

Some of the ages that were generated had some "17 and 115 year old" officers making some Terry Stops, but after checking at what age officers can actually carry out their duties is at the age of 21. It was also highly unlikely that there were 115 year old officers making stops so I only inlcuded data in my DataFrame with officers aged 21-69 years old.

The weapon_type column was binned to be either True (1) or False (0), if they had a weapon or not.

The target (stop_resolution), was initially binned to bee boolean as in if an arrest was made or not, but after some careful consideration, it was decided to bin it as whether the stop resulted in a "major" or "minor" consequence.

Major consequence (1):

- Arrest
- Offense Report
- Referred for Prosecution

Minor consequence (0):
- Field Contact
- Citation / Infraction


### Choosing Features ###

Some features were selected to produce the predictive model:

- 'sub_age_group'
- 'officer_gender'
- 'officer_race'
- 'sub_perceived_race'
- 'sub_perceived_gender'
- 'frisk_flag'
- 'call_type'


## Model 1 ##

The base model was created using LogisiticRegression with no hyperparamenters changed. Some predictions were generated and scored using the F1 scoring metric. The recall score and precision score were also generated as the scores are the **Harmonic Mean of Precision and Recall**. 

The following were the evaluation metrics generated:

- F1 score for training data: 0.8802693777115845

- F1 score for testing data: 0.8781435090785512

- Recall score for training data: 0.9016980631467233

- Recall score for testing data: 0.8972222222222223

- Precision score for training data: 0.8598355471220747

- Precision score for testing data: 0.8598592888381822


**Confusion Matrix**
![image1](https://github.com/edgarbarr1/terry_stops/blob/main/images/base_model_cf.png)


Overall solid scores. There was no sign of the model overfitting or underfitting. However, it was important to see if the model could be improved to give a better score on the metric.

**ROC Curve**

![image2](https://github.com/edgarbarr1/terry_stops/blob/main/images/ROC_curve_1.png)

AUC score: 0.9014

Overall good evaluating metric. Let's see if we can improve this.


## Model 2 ##

The second model was made with synthetic data using `SMOTE()`. After transforming my data using SMOTE, I fit it into my base LogisticRegression model, however the F1 score decreased by a bit while the AUC score increased for just a tiny bit.

The following were the evaluation metrics generated:

- F1 score for training data: 0.8518604113704645

- F1 score for testing data: 0.8654663160024275

- Recall score for training data: 0.855730963120191

- Recall score for testing data: 0.8488095238095238

- Precision score for training data: 0.8480247157036744

- Precision score for testing data: 0.8827899298390425


**Confusion Matrix**

![image3](https://github.com/edgarbarr1/terry_stops/blob/main/images/model_2_cf.png)

Good F1 score but not better than the original base model. 

**ROC Curve**

![image4](https://github.com/edgarbarr1/terry_stops/blob/main/images/ROC_curev_model_2_cf.png)


## Model 3 ##

The third model still had as basis LogisitcRegression but this time around, the penalty of the model was adjusted to have an `l1 ` instead of the standard `l2`. The results are as follow:

The following were the evaluation metrics generated:

- F1 score for training data: 0.8523031203566122

- F1 score for testing data: 0.8653515427415276

- Recall score for training data: 0.8560626160785354

- Recall score for testing data: 0.8486111111111111

- Precision score for training data: 0.8485765007561312

- Precision score for testing data: 0.882765737874097


**Confusion Matrix**

![image5](https://github.com/edgarbarr1/terry_stops/blob/main/images/model_3_cf.png)

Not much of a change compared to the couple of previous models in regards to the evaluation metrics.

**ROC Curve**

![image6](https://github.com/edgarbarr1/terry_stops/blob/main/images/ROC_curve_model_3_cf.png)

Again, not much change.


## Model 4 ##

This model was based on LogisticRegression and stayed at the default `l2` penalty. A loop was ran using different C-values on the LogisticRegression hyperparameter. It was found that the best C-value was 1e-1 because it was the C-value that least deviated the F1 score from the training data vs the testing data.

Then, using the C-value of 1e-1, the model ran through various class weights in a loop to see teh effect of the various class weights in the performance of the model. The ROC curve visualization below was the end result.

**ROC Curve**
![image8](https://github.com/edgarbarr1/terry_stops/blob/main/images/ROC_curves_c_values.png)

Based on the visualization not much needed to change in terms of class weights. The best producing model was the class weight of 2 to 1 and even then, the AUC score had a negligible increase that keeping the class weight as default (`None`) or leaving it as `balanced` would've been overall the best choice.


## Model 5 ##

After changing different hyperparameters in the LogisitcRegression it was time to move to the KNN classifier. In this model the base model was used to fit to the data. The data was scaled using the StandardScaler and transformed using OneHotEncoder. Then the data was fit to the model.

The following were the evaluation metrics generated:

- F1 score for training data: 0.8956018670698835

- F1 score for testing data: 0.8621845425867508

- Recall score for training data: 0.9036216503051208

- Recall score for testing data: 0.8676587301587302

- Precision score for training data: 0.8877231851948391

- Precision score for testing data: 0.8567789968652038


**Confusion Matrix**

![image9](https://github.com/edgarbarr1/terry_stops/blob/main/images/KNN_cf.png)


## Model 6 ##

This model was fit using the previously transformed data with StandardScaler and OneHotEncoder but first using a function to find the best k that would have the best evaluating metrics. After running it, it was found that the best k was of 47. This was then applied to the KNN model and fit to the data.

The following were the evaluation metrics generated:

- F1 score for training data: 0.8797674794122397

- F1 score for testing data: 0.8783365570599614

- Recall score for training data: 0.903488989121783

- Recall score for testing data: 0.9009920634920635

- Precision score for training data: 0.8572597394423815

- Precision score for testing data: 0.8567924528301887


**Confusion Matrix**

![image11](https://github.com/edgarbarr1/terry_stops/blob/main/images/KNN_bestK_cf.png)


## Final Model ##

Based on the evaluation metrics, it was found that Model 1 had the best F1 score and therfore best run model. Although Model 6 had on par evaluating metrics, its not so quick runtime held it back from being considered as the Final model of this data. As a rule of precaution, the first train test split (X_train_all, X_test_all, y_train_all, y_test_all) had not been fit to the model and therefore was chosen to be a final test to the LogisticRegression model to verify the evaluating metrics.

The following were the evaluation metrics generated:

- F1 score for training data: 0.8797245860304992

- F1 score for testing data: 0.8795189347841649

- Recall score for training data: 0.9019188705508053

- Recall score for testing data: 0.9019233592717663

- Precision score for training data: 0.8585963749940845

- Precision score for testing data: 0.8582006146968427

Almost exactly if not better performance metrics in regards to the F1 score.

**Confusion Matrix**

![image13](https://github.com/edgarbarr1/terry_stops/blob/main/images/final_model_tts2.png)

**ROC Curve**

![image14](https://github.com/edgarbarr1/terry_stops/blob/main/images/ROC_curve_final_model.png)

After verifying with never before fit data, the whole dataset was fit to the base LogisticRegression model and the following were the results:

- F1 score for final model: 0.8792906903103229
- Recall score for final model: 0.9023285178445426
- Precision score for final model: 0.8573999576540334

**Final Confusion Matrix**

![image15](https://github.com/edgarbarr1/terry_stops/blob/main/images/full_data_cf.png)


# Conclusion #

Overall, with a .90 AUC score and a .88 F1 score we have a solid performing model for the dataset. One of the main reasons it is believed why the data performed so well with the base model was because of the binning of the target variable as well as the simplification done in some of the features during the EDA process of the project.

Some lingering quesitons remain regarding the performance of the model. With more time more models would've been fit including a decision tree that could've potentially improved the performance and evaluation metric of the model. 

It would also be noteworthy to know what specific features contributed the most to the ableness to predict whether a person would've had a "major" or "minor" consequence because of a Terry Stop.