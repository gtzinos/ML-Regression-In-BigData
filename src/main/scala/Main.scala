import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql
import org.apache.spark.sql.functions._
object Main {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  def token_tfidf(dataframe: Dataset[Row], columns: Array[String]): sql.DataFrame  = {
    var completeDF:sql.DataFrame = dataframe
    for(row <- columns) {
      // Create tf-idf features
      val tokenizer = new Tokenizer().setInputCol(row).setOutputCol("token_" + row)
      completeDF = tokenizer.transform(completeDF)
      //delete old column
      completeDF = completeDF.drop(row)
    }

    for(row <- columns) {
      val hashingTF = new HashingTF().setInputCol("token_" + row).setOutputCol("hash_" + row)
      completeDF = hashingTF.transform(completeDF)
      //delete old column
      completeDF = completeDF.drop("token_" + row)
    }

    for(row <- columns) {
      val idf = new IDF().setInputCol("hash_" + row).setOutputCol("feature_" + row)
      val idfM = idf.fit(completeDF)
      completeDF = idfM.transform(completeDF)
      //delete old column
      completeDF = completeDF.drop("hash_" + row)
    }

    completeDF
  }
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

    val fullSchemaDF = ss.sql("Select tr.product_title, tr.search_term, tr.relevance as label" +
      " , IFNULL(attr.name, '') as name, IFNULL(attr.value, '') as value, pr.product_description from trainDF tr left join attributesDF attr on" +
      " tr.product_uid == attr.product_uid left join productsDescDF pr on tr.product_uid == pr.product_uid")

    fullSchemaDF.repartition(4)

    val tokenizedDF = token_tfidf(fullSchemaDF, Array("product_title", "product_description", "name", "search_term", "value"))

    import org.apache.spark.sql.types._
    val completeDF = tokenizedDF.withColumn("label", 'label cast DoubleType)

    //completeDF.show()
    /* ======================================================= */
    /* ================== CLASSIFICATION ===================== */
    /* ======================================================= */

    println("BEFORE TRAINING")
    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = completeDF.randomSplit(Array(0.7, 0.3), seed = 1234L)

    //trainingData.write.json("./data/trainSaved.json")
    //testData.write.json("./data/testSaved.json")


    //val train = ss.read.option("header", "true").csv("./data/trainSaved.csv")
    //val test = ss.read.option("header", "true").csv("./data/testSaved.csv")

    val LRmodel = new LinearRegression()
      .setMaxIter(10000)
      .setRegParam(0.1)
      .setElasticNetParam(0.0)
      .setFeaturesCol("feature_product_description")
      .fit(trainingData)

    val predictionsLR = LRmodel.transform(testData)

    predictionsLR.printSchema()
    predictionsLR.show(10)

    // Evaluate the model by finding the accuracy
    val evaluatorNB = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracyLR = evaluatorNB.evaluate(predictionsLR)
    println("Accuracy of Logistic Regression: " + accuracyLR)
  }
}
