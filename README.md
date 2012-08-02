Classifier
==========

<h4>Automated Document Classificatier using Complementary NaiveBayes Algorithm.</h4>

<b>Trainer.java</b>
     Takes unprocessed data set and produces processed dataset as suitable for Mahout file format. Responsible for training Complementary Naive bayes algorithm and build a statistical model. 

<b>Classifier.java</b>
       Takes an unclassified data directory and classifies the documents. Creates separate subdirectories for each category and writes the files onto the directory. 


<b>Setting Up Parameters in settings.properties file</b>
<i>Bayesparameters</i>

Gramsize=2              // Ngram size
Algorithm=cbayes      // our classification  algorithm
DefaultCategory=unknown  // Default Category
DataSource=hdfs          // Hadoop File System
Encoding=UTF-8        // Unicode
Alpha=1.0                   //Smoothing parameter


<b>For Trainer.java</b>

TrainSet=/home/developer/dataset_rev/freshrevs/train/              // training set location which containing subdirectories of each category
ProcessedSet=/home/developer/dataset_rev/freshrevs/processedTrain/          // Processed Output Directory

<b>For Classifier.java</b>

ModelPath=/home/developer/dataset_rev/freshrevs/model/                    //    Path to store and retrieve Model 
IpDirPath=/home/developer/dataset_rev/freshrevs/test/pos/                    //   Unclassifed data set
OpDirPath=/home/developer/dataset_rev/freshrevs/classified/              //   Path to store classified documents

