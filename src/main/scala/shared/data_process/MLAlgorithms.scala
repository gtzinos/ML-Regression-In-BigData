package shared.data_process

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression._
import org.apache.spark.sql
import org.apache.spark.sql.Row
import shared.data_process.MLAlgorithms
import scala.collection.mutable.ListBuffer

class MLAlgorithms(dataset: sql.Dataset[Row], trainPercentage:Double, testPercentage: Double, fullSummary: Boolean) {
  //Split and declare train/ test data on create
  val (trainingData, testData) = splitData(dataset, trainPercentage, testPercentage)
  var allrmse  = new ListBuffer[String]()


  //Split dataset
  private def splitData(dataset: sql.Dataset[Row], trainPercentage:Double, testPercentage: Double) = {

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = dataset.randomSplit(Array(trainPercentage, testPercentage), seed = 1234L)

    (trainingData, testData)
  }

  def RunLinearRegression(maxIter: Integer, regParam: Double, elasticNetParam: Double) = {

    //Train dataset with linear regression algorithm
    val lrModel = new LinearRegression()
      .setMaxIter(maxIter)
      .setRegParam(regParam)
      .setElasticNetParam(elasticNetParam)
      .setFeaturesCol("all_features")
      .fit(trainingData)

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary

    val predictions = lrModel.transform(testData)

    if(fullSummary) {
      println(s"numIterations: ${trainingSummary.totalIterations}")
      println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")

      trainingSummary.residuals.show()

      println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
      println(s"r2: ${trainingSummary.r2}")

      predictions.select("prediction", "label").show(50, false)
    }

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    
    allrmse += "Linear Regression: " + String.valueOf(rmse) + " maxItem=" + String.valueOf(maxIter) + " regParam=" + String.valueOf(regParam) + " elasticNetParam=" + String.valueOf(elasticNetParam)
    
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)
  }

  def RunDecisionTree(maxCategories: Integer = 4) = {
    val featureIndexer = new VectorIndexer()
      .setInputCol("all_features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(maxCategories)
      .fit(dataset)

    // Train a DecisionTree model.
    val dt = new DecisionTreeRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")

    // Chain indexer and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, dt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)


    allrmse += "Decision Tree: " + String.valueOf(rmse) + " masxCategorie=" + String.valueOf(maxCategories)
    
    if(fullSummary) {
      // Select example rows to display.
      predictions.select("prediction", "label", "features").show(5)

      val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
      println("Learned regression tree model:\n" + treeModel.toDebugString)
    }
  }

  def RunRandomForestRegressor(numTrees: Int = 20, maxDepth: Int = 5) = {
    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setNumTrees(numTrees)
      .setMaxDepth(maxDepth)
      .setLabelCol("label")
      .setFeaturesCol("all_features")

    // Train model. This also runs the indexer.
    val model = rf.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    allrmse += "Random Forest: " + String.valueOf(rmse) + " numTrees=" + String.valueOf(numTrees) + " maxDepth=" + String.valueOf(maxDepth)
    
    if (fullSummary) {
      val rfModel = model.asInstanceOf[RandomForestRegressionModel]
      println("Learned regression forest model:\n" + rfModel.toDebugString)
    }

  }

  def RunGBTRegressor(maxCategories: Integer=4) = {
    val featureIndexer = new VectorIndexer()
      .setInputCol("all_features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(maxCategories)
      .fit(dataset)

    // Train a GBT model.
    val gbt = new GBTRegressor()
      .setLabelCol("label")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)

    // Chain indexer and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, gbt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    if(fullSummary) {
      // Select example rows to display.
      predictions.select("prediction", "label", "features").show(5)

      val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
      println("Learned regression GBT model:\n" + gbtModel.toDebugString)
    }

    // Select (prediction, true label) and compute test error.
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

    allrmse += "GBT Regression: " + String.valueOf(rmse) + " maxCategories=" + String.valueOf(maxCategories)
  }
  
  def printErrors() = {
    allrmse.foreach(println)
  }
}
