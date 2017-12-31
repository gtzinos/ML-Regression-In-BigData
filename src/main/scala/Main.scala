import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.functions._

object Main {

  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {

    // Create the spark session first
    val ss = SparkSession.builder().master("local").appName("tfidfApp").getOrCreate()


    import ss.implicits._ // For implicit conversions like converting RDDs to DataFrames


    val trainFileName = "./data/train.csv"
    val attributesFileName = "./data/attributes.csv"
    val descriptionsFileName = "./data/product_descriptions.csv"

    val trainDF = ss.read.option("header", "true").csv(trainFileName).createOrReplaceTempView("trainDF")
    val attributesDF = ss.read.option("header", "true").csv(attributesFileName).createOrReplaceTempView("attributesDF")
    val productsDescDF = ss.read.option("header", "true").csv(descriptionsFileName).createOrReplaceTempView("productsDescDF")

    ss.sql("Select tr.*, attr.name, attr.value, pr.product_description from trainDF tr left outer join attributesDF attr on" +
      " tr.product_uid == attr.product_uid left outer join productsDescDF pr on tr.product_uid == pr.product_uid").printSchema()


/*
    // Rename the columns of the dataframe
    val newColumnNames = Seq("rId", "rText", "rLabel")
    val renamedDF = basicDF.toDF(newColumnNames: _*)
    renamedDF.printSchema()
    renamedDF.select("rId", "rText", "rLabel").take(5).foreach(println)

    // Set the number of partitions
    val reviewsDF = renamedDF
    reviewsDF.repartition(4)

    // Create tf-idf features
    val tokenizer = new Tokenizer().setInputCol("rText").setOutputCol("rWords")
    val wordsDF = tokenizer.transform(reviewsDF)
    wordsDF.printSchema()

    val hashingTF = new HashingTF().setInputCol("rWords").setOutputCol("rRawFeatures") //.setNumFeatures(20000)
    val featurizedDF = hashingTF.transform(wordsDF)
    featurizedDF.printSchema()

    val idf = new IDF().setInputCol("rRawFeatures").setOutputCol("rFeatures")
    val idfM = idf.fit(featurizedDF)
    val completeDF = idfM.transform(featurizedDF)
    completeDF.printSchema()
    completeDF.select("rId", "rFeatures", "rWords").take(3).foreach(println)

    val udf_toDouble = udf((s: String) => if (s == "positive") 1.0 else 0.0)
    //val udf_toDense = udf((v: SparseVector) => v.toDense)

    val newDF = completeDF.select($"rId", $"rText", $"rWords", $"rRawFeatures", $"rFeatures", $"rLabel", udf_toDouble($"rLabel").as("rrLabel"))
    newDF.printSchema()
    newDF.select("rId", "rLabel", "rrLabel").take(100).foreach(println)


    /* ======================================================= */
    /* ================== CLASSIFICATION ===================== */
    /* ======================================================= */


    // Make sure the columns <features> and <label> exist
    val df = newDF.select("rId", "rFeatures", "rrLabel").withColumnRenamed("rFeatures", "features").withColumnRenamed("rrLabel", "label")

    println("BEFORE TRAINING")

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3), seed = 1234L)

    /* ===============================================================================  */
    /* Classification example using Naive Bayes Classifier                              */
    /* ===============================================================================  */

    val NBmodel = new NaiveBayes().fit(trainingData)

    val predictionsNB = NBmodel.transform(testData)
    predictionsNB.printSchema()
    //predictionsNB.take(100).foreach(println)
    //predictionsNB.select("label", "prediction").show(100)
    predictionsNB.show(10)

    // Evaluate the model by finding the accuracy
    val evaluatorNB = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracyNB = evaluatorNB.evaluate(predictionsNB)
    println("Accuracy of Naive Bayes: " + accuracyNB)


    /* ===============================================================================  */
    /* Classification example using Logistic Regression Classifier                     */
    /* ===============================================================================  */

    val LRmodel = new LogisticRegression()
      .setMaxIter(10000)
      .setRegParam(0.1)
      .setElasticNetParam(0.0)
      .fit(trainingData)

    val predictionsLR = LRmodel.transform(testData)
    predictionsLR.printSchema()
    predictionsLR.show(10)

    val accuracyLR = evaluatorNB.evaluate(predictionsLR)
    println("Accuracy of Logistic Regression: " + accuracyLR)
*/
  }
}
