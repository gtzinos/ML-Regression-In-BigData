package shared.data_process

import org.apache.spark.sql.SparkSession

class ImportDataset {

  def getFileName(path: String) = {
    val paths: Array[String] = path.split("/")
    val fileName = paths(paths.length-1).split('.')(0)

    fileName
  }

  def importDatasets(session: SparkSession, datasets: Array[String]) = {
    for(dataset: String <- datasets) {
      val fileName = getFileName(dataset)
      session.read.option("header", "true").csv(dataset).createOrReplaceTempView(fileName)
    }
  }
}
