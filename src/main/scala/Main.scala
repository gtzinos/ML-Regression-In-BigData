import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.regression._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.SparkConf
import shared.data_process.{ImportDataset, MLAlgorithms, PreProcessing}
object Main {
  Logger.getLogger("org").setLevel(Level.OFF)
  Logger.getLogger("akka").setLevel(Level.OFF)

  //Join 3 datasets (Products, Descriptions, Attributes)
  def joinDatasets(session: SparkSession) : Dataset[Row]= {
    val fullSchemaDF = session.sql("Select concat(tr.product_title, tr.search_term, pr.product_description) as text, tr.relevance as label" +
      //  " , IFNULL(attr.name, '') as name, IFNULL(attr.value, '') as value, pr.product_description from train tr left join attributes attr on" +
      " from train tr left join product_descriptions pr on tr.product_uid == pr.product_uid")

    //Partition dataframe
    fullSchemaDF.repartition(4)

    fullSchemaDF
  }

  def saveDataFrame(frame: sql.DataFrame) = {
    /*
      Save and restore the merged dataset
    */

    //TODO: save dataset
    //trainingData.write.json("./data/trainSaved.json")
    //testData.write.json("./data/testSaved.json")

    //TODO: restore dataset
    //val train = ss.read.option("header", "true").csv("./data/trainSaved.csv")
    //val test = ss.read.option("header", "true").csv("./data/testSaved.csv")
  }

  //Main function
  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().set("spark.driver.maxResultSize", "8g")

    // Create the spark session first
    val ss = SparkSession.builder().config(conf).master("local").appName("tfidfApp").getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    import ss.implicits._

    val preprocessing = new PreProcessing()
    val importDataset = new ImportDataset()

    //Datasets files names
    val trainFileName = "./data/train.csv"
    val attributesFileName = "./data/attributes.csv"
    val descriptionsFileName = "./data/product_descriptions.csv"

    //Import datasets and create temp view
    importDataset.importDatasets(ss, Array(trainFileName, attributesFileName, descriptionsFileName))

    val fullSchemaDF = joinDatasets(ss)

    //Tokenization and tfidf for string features
    val tokenizedDF = preprocessing.text_preprocessing(fullSchemaDF, Array("text"), 16000)

    //To cast to Double
    import org.apache.spark.sql.types._
    //Cast labels to double from string
    val completeDF = tokenizedDF.withColumn("label", 'label cast DoubleType)

    //Save the results to restart from this point
    saveDataFrame(completeDF)

    /* ======================================================= */
    /* ================== Regression ===================== */
    /* ======================================================= */

    println("BEFORE TRAINING")

    val mlAlgorithms = new MLAlgorithms(completeDF, 0.7, 0.3, false)

    //Predictions from Linear Regression
    //mlAlgorithms.RunLinearRegression(10000, 0.2, 0.05)
    
    mlAlgorithms.RunRandomForestRegressor(5, 5)
    mlAlgorithms.RunRandomForestRegressor(10, 5)
    mlAlgorithms.RunRandomForestRegressor(15, 5)
    mlAlgorithms.RunRandomForestRegressor(20, 5)
    mlAlgorithms.RunRandomForestRegressor(20, 10)

    mlAlgorithms.printErrors()

    mlAlgorithms.RunDecisionTree(4)
    mlAlgorithms.RunDecisionTree(5)
    mlAlgorithms.RunDecisionTree(6)
    mlAlgorithms.RunDecisionTree(7)
    mlAlgorithms.RunDecisionTree(8)

    mlAlgorithms.printErrors()

    mlAlgorithms.RunGBTRegressor(4)
    mlAlgorithms.RunGBTRegressor(5)
    mlAlgorithms.RunGBTRegressor(6)
    mlAlgorithms.RunGBTRegressor(7)
    mlAlgorithms.RunGBTRegressor(8)

    mlAlgorithms.printErrors()

    //Predictions from RandonForest
    //mlAlgorithms.RunRandomForestRegressor()

    //Predictions from Decision Tree
    //mlAlgorithms.RunDecisionTree()

    //Predictions from GBT Regressor
    //mlAlgorithms.RunGBTRegressor()
   }
}
