# Public Project

## Executive Summary

Using a seminal study on wine from the Vinho Verde region of Portugal, this project looked to create a predictive model capable of classifying wine quality based on physiochemical properties, to help stakeholders identify key quality factors in wine. 

A logistic regression model was selected as the baseline algorithm, as a test of the linear relationship between the features and wine quality. 

The model demonstrated decent recall for high and low quality wines (0.6 and 0.73) but poor precision, indicating a high rate of false positives, medium quality wines incorrectly being flagged as high quality.

Future iterations could look to use a more advanced model, such as XGBoost to hyper define the parameters, or improve the binning strategy by expanding the options or including the original score boundaries. 

## Data Infrastructure & Tools

The project was developed using cloud-compatible data science tools, that would ensure scalability and reproducibility, which should help to improve both the reliability and the interpretability of the models. These tools included, Google Colab, chosen for its access to GPU and the cost, as little money was available to execute the project, and Colab grants easy access to every user, simply needing a Google account. Alternatives such as Deepnote could have been utilised, but the familiarity and accessibility of Colab made it the ideal choice. Future iterations of this project could look to move the notebooks into a wider ecosphere such as Azure/Fabric to automate ingestion and engineering, and google colab would easily allow this.

The choice of libraries to use within colab was driven by ease-of-use and scalability, and so Scikit-learn was chosen as the main library to allow for algorithmic modelling, namely logistic regression, alongside seaborn for visualization. In future, within Azure, these libraries could be dropped in favour of native PowerBI visualisations, which would allow for more interactivity. Further, and if money was less of an object, some AI tools that help with simple natural language processing to autogenerate more interesting and interpretable visuals, such as CoPilot or Julius AI.3

Finally, the dataset itself was sourced from Kaggle (Kaggle, n.d.), where users have merged the seminal UCI study (Cortez, 2009), a data set of white and red wines from the Vinho Verde area of Portugal. The benefits of using a public dataset from Kaggle include the ability to utilise the API function to automate ingestion, and avoidance and cost of licensing the dataset, or ethical issues of scraping a dataset from the internet, such as introducing bias, legal ambiguity of ownership of data and potential privacy violations. 

## Data Engineering

A pipeline was initialised in Colab using the Kaggle API to ensure the dataset remained up-to-date on each run. Standard libraries were loaded for analysis. The API approach future-proofs the ingestion, howver there is a risk of the publicly sourced dataset being moved or decommissioned. 


### Libraries Loaded
![Libraries loaded](images/Fig%201%20Libraries%20loaded.png)
<!-- This us a comment-->
