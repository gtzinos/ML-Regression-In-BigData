package shared.data_process

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql
import org.apache.spark.sql.Row
import shared.data_process.MLAlgorithms

class MLAlgorithms {

  private def splitData(dataframe: sql.Dataset[Row]) = {

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = dataframe.randomSplit(Array(0.8, 0.2), seed = 1234L)

    (trainingData, testData)
  }

  def RunLinearRegression(dataframe: sql.DataFrame) = {

    val (trainingData, testData) = splitData(dataframe)

    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
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

    val rfModel = model.asInstanceOf[RandomForestRegressionModel]
    println("Learned regression forest model:\n" + rfModel.toDebugString)

    /*
        //Train dataset with linear regression algorithm
        val lrModel = new LinearRegression()
          .setMaxIter(10000)
          .setRegParam(0.2)
          .setElasticNetParam(0.05)
          .setFeaturesCol("all_features")
          .fit(trainingData)


        // Summarize the model over the training set and print out some metrics
        val trainingSummary = lrModel.summary
        println(s"numIterations: ${trainingSummary.totalIterations}")
        println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
        trainingSummary.residuals.show()
        println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
        println(s"r2: ${trainingSummary.r2}")

        val predictions = lrModel.transform(testData)
        predictions.select("prediction", "label").show(50, false)

        val evaluator = new RegressionEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
          .setMetricName("rmse")

        val rmse = evaluator.evaluate(predictions)
        println("Root Mean Squared Error (RMSE) on test data = " + rmse)
    */


    /*
        // Print the coefficients and intercept for linear regression
        println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

        // Summarize the model over the training set and print out some metrics
        val trainingSummary = lrModel.summary
        //println(s"numIterations: ${trainingSummary.totalIterations}")
        //println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
        trainingSummary.residuals.show()
        println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
        println(s"r2: ${trainingSummary.r2}") */



    /*val predictionsLR = LRmodel.transform(testData)
    predictionsLR.printSchema()
    predictionsLR.show(10)
    // Evaluate the model by finding the accuracy
    val evaluatorNB = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracyLR = evaluatorNB.evaluate(predictionsLR)
    println("Accuracy of Logistic Regression: " + accuracyLR)*/

  }
}
