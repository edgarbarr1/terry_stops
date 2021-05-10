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
![image1]()


Overall solid scores. There was no sign of the model overfitting or underfitting. However, it was important to see if the model could be improved to give a better score on the metric.

