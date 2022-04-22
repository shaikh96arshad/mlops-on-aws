**Project Architecture**
![Project Architecture](https://github.com/shaikh96arshad/AWS-Projects/blob/main/Bank%20Personal%20Loan%20Modelling/images/pipeline_img.png)

**Context**:

This case is about a bank (Thera Bank) which has a growing customer base. Majority of these customers are liability customers (depositors) with varying size of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with minimal budget.


**The components of the reference architecture diagram are:**

1.A secured development environment was implemented using an Amazon SageMaker Notebook Instance deployed to a custom virtual private cloud (VPC), and secured by implementing security groups and routing the notebook’s internet traffic via the custom VPC.

2.Also, the development environment has two Git repositories (AWS CodeCommit) attached: one for the Exploratory Data Analysis (EDA) code and the other for developing the custom Amazon SageMaker Docker container images.

3.An ML CI/CD pipeline made up of three sub-components:
Data validation step implemented using AWS Lambda and triggered using AWS CodeBuild.
Model training/retraining pipeline implemented using Amazon SageMaker Pipelines (pipeline-as-code) and executed using CodeBuild.
Model deployment pipeline that natively supports model rollbacks was implemented using AWS CloudFormation.
Finally, AWS CodePipeline is used to orchestrate the pipeline.

4.A CI/CD pipeline for developing and deploying the custom Amazon SageMaker Docker container image. This pipeline automatically triggers the ML pipeline when you successfully push a new version of the SageMaker container image, providing the following benefits:
Developers and data scientists can thoroughly test and get immediate feedback on the ML pipeline’s performance after publishing a new version of the Docker image. This helps ensure ML pipelines are adequately tested before promoting them to production.
Developers and data scientists don’t have to manually update the ML pipeline to use the latest version of the customized Amazon SageMaker image when working on the develop git branch. They can branch off the develop branch if they want to use an older version or start developing a new version, which they will merge back to develop branch once approved.

5.A model monitoring solution implemented using Amazon SageMaker Model Monitor to monitor the production models’ quality continuously.

6.This provides monitoring for the following: data drift, model drift, the bias in the models’ predictions, and drifts in feature attributes. You can start with the default model monitor, which requires no coding.

7.A model retraining implementation that is based on the metric-based model retraining strategy. There are three main retaining strategies available for your model retraining implementation:
Scheduled: This kicks off the model retraining process at a scheduled time and can be implemented using an Amazon EventBridge scheduled event.
Event-driven: This kicks off the model retraining process when a new model retraining dataset is made available and can be implemented using an EventBridge event.
Metric-based: This is implemented by creating a Data Drift CloudWatch Alarm (as seen in Figure 1 above) that triggers your model retraining process once it goes off, fully automating your correction action for a model drift.
A data platform implemented using Amazon S3 buckets with versioning enabled.