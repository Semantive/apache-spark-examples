package com.semantive.sensors

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

trait Sensors {

  def numClasses = 6

  def categoricalFeatures = Map[Int, Int]()

  def prepareData(sc: SparkContext) = {
    val trainX = prepareRdd(sc.textFile("/sensors-dataset/data/train/X_train.txt"))
    val trainY = prepareRdd(sc.textFile("/sensors-dataset/data/train/y_train.txt"))

    val testX = prepareRdd(sc.textFile("/sensors-dataset/data/test/X_test.txt"))
    val testY = prepareRdd(sc.textFile("/sensors-dataset/data/test/y_test.txt"))

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

  private def prepareRdd(rdd: RDD[String]) =
    rdd.map(l => l.split("\\s+").filterNot(_.isEmpty).map(_.toDouble)).zipWithIndex().map(_.swap)

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
