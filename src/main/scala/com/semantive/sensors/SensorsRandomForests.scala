package com.semantive.sensors

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object SensorsRandomForests extends Sensors with App {

  val featureSubsetStrategy = "auto"
  val maxDepth = 30

  val sc = new SparkContext(new SparkConf().setAppName("SensorsRandomForests").setMaster("local[4]"))
  val (train, test, validation) = prepareData(sc)

  val evaluations = for {
    impurity <- Array("gini", "entropy")
    maxBins <- Array(25, 50, 100, 200, 300)
    numTrees <- Array(50, 100, 200)
  } yield {
    println(s" $impurity $maxBins $numTrees")
    val model = RandomForest.trainClassifier(
      train,
      numClasses,
      categoricalFeatures,
      numTrees,
      featureSubsetStrategy,
      impurity,
      maxDepth,
      maxBins)
    (model, (impurity, maxBins, numTrees), evaluateModel(getMetrics(model, validation)).map(_.sum).sum)
  }

  val (model, (impurity, maxBins, numTrees), _) = evaluations.maxBy(_._3)
  println(s"Selected impurity: $impurity, selected maxBins: $maxBins, selected numTrees: $numTrees")
  val metrics = getMetrics(model, test)
  println(metrics.confusionMatrix)
  evaluateModel(metrics).foreach(println)

  def getMetrics(model: RandomForestModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example => (model.predict(example.features), example.label))
    new MulticlassMetrics(predictionsAndLabels)
  }
}
