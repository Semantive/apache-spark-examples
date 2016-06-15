package com.semantive.sensors

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

trait Sensors {

  def numClasses = 6
  def categoricalFeatures = Map[Int, Int]()

  def prepareData(sc: SparkContext) = {
    val trainX = sc.textFile("/sensors-dataset/data/train/X_train.txt")
      .map(l => l.split("\\s+").filterNot(_.isEmpty).map(_.toDouble)).zipWithIndex().map(_.swap)
    val trainY = sc.textFile("/sensors-dataset/data/train/y_train.txt")
      .map(l => l.split(" ").filterNot(_.isEmpty).map(_.toDouble)).zipWithIndex().map(_.swap)

    val testX = sc.textFile("/sensors-dataset/data/test/X_test.txt")
      .map(l => l.split("\\s+").filterNot(_.isEmpty).map(_.toDouble)).zipWithIndex().map(_.swap)
    val testY = sc.textFile("/sensors-dataset/data/test/y_test.txt")
      .map(l => l.split(" ").filterNot(_.isEmpty).map(_.toDouble)).zipWithIndex().map(_.swap)

    val train = trainX.join(trainY).map {
      case (i, (values, labels)) =>
        LabeledPoint(labels.head - 1, Vectors.dense(values))
    }.cache()
    val testAndValidation = testX.join(testY).map {
      case (i, (values, labels)) =>
        LabeledPoint(labels.head - 1, Vectors.dense(values))
    }.cache()
    val Array(test, validation) = testAndValidation.randomSplit(Array(0.8, 0.2))
    (train, test, validation)
  }

  def evaluateModel(metrics: MulticlassMetrics) = {
    (0 until 6).map(
      cat => List(
        metrics.precision(cat),
        metrics.recall(cat),
        metrics.truePositiveRate(cat),
        metrics.falsePositiveRate(cat),
        metrics.fMeasure(cat))
    )
  }
}
