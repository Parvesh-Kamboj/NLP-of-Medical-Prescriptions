# NLP-of-Medical-Prescriptions
NLP of Medical prescriptions and then applying regression models to predict base scores for patients
Few important things about code:
1. I have transformed “name_of_drug” and “use_case_for_drug” using target based encoding along with Laplace smoothing. Laplace smoothing has a weight parameter which can also be hyper-tunned.
2. The “use_case_for_drug” has some weird texts, we can either replace the weird text with the new category(example “NA”) or we can actually replace them with the mode of “use_case_for_drug” grouped over “ name_of_drug”. For simplicity I have utilised the first method.  
3. I have transformed “reviews_by_patient” by using average word2vec model after cleaning it.
4. I have transformed “drug_approved_by_UIC” into a column with logic of (date is equivalent to days from the current date). Hence, dropped “drug_approved_by_UIC” after transformation. 
5. “patient_id” is just a unique key here so dropping it directly
6. I have utilised support vector machine regressor as number of dimensions after transformation are large. We can utilise any algorithm like deep neural network for complex relations or even simply linear regression as a start point.
7. We can skip first 3 steps if we have utilised catboost algorithm.
8. The given problem though seems like a regression problem but could also be solved as classification problem by binning it over “base_score”
