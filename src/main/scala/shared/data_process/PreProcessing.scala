package shared.data_process

import org.apache.spark.ml.feature._
import org.apache.spark.sql
import org.apache.spark.sql.{Dataset, Row}

class PreProcessing {

  def removeStopWords(dataframe: sql.Dataset[Row], inputColumn: String, outputColumn:String): sql.Dataset[Row] = {
    val clear = new StopWordsRemover()
      .setCaseSensitive(false)
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))
      .setInputCol(inputColumn)
      .setOutputCol(outputColumn)
      .transform(dataframe)

    val complete = clear.drop(inputColumn)

    complete
  }

  def tokenized(dataframe: sql.Dataset[Row], inputColumn: String, outputColumn:String): sql.Dataset[Row] = {
    val tokenizer = new Tokenizer().setInputCol(inputColumn).setOutputCol(outputColumn)

    val transormed = tokenizer.transform(dataframe)
    //delete old column
    val tokenized = transormed.drop(inputColumn)

    tokenized
  }

  def tf(dataframe: sql.Dataset[Row], inputColumn: String, outputColumn:String, numFeatures: Int): sql.Dataset[Row] = {
    val hashingTF = new HashingTF().setInputCol(inputColumn).setOutputCol(outputColumn)setNumFeatures(numFeatures)

    val hashed = hashingTF.transform(dataframe)
    //delete old column
    val complete = hashed.drop(inputColumn)

    complete
  }

  def idf(dataframe: sql.Dataset[Row], inputColumn: String, outputColumn:String) = {
    val idf = new IDF().setInputCol(inputColumn).setOutputCol(outputColumn)
    val idfM = idf.fit(dataframe)

    val transformed = idfM.transform(dataframe)
    //delete old column
    val complete = transformed.drop(inputColumn)

    complete
  }

  //Tokenization and tfidf for string features
  def text_preprocessing(dataframe: Dataset[Row], columns: Array[String], numFeatures: Int): sql.DataFrame  = {
    //global dataframe
    var completeDF:sql.DataFrame = dataframe

    //Remove stop words

    for(row <- columns) {
      completeDF = tokenized(completeDF, row, "token_" + row)
      completeDF = removeStopWords(completeDF, "token_" + row, "stopwords_" + row)
      completeDF = tf(completeDF, "stopwords_" + row, "hash_" + row, numFeatures)
      completeDF = idf(completeDF, "hash_" + row, row)
    }

    //Create all features column
    val assembler = new VectorAssembler().setInputCols(columns).setOutputCol("all_features")
    completeDF = assembler.transform(completeDF)

    //Return dataframe
    completeDF
  }
}
