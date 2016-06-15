package com.semantive.sensors

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object SensorsTrees extends Sensors with App {

  val maxDepth = 30

  val sc = new SparkContext(new SparkConf().setAppName("SensorsTrees").setMaster("local[4]"))
  val (train, test, validation) = prepareData(sc)

  val evaluations = for {
    impurity <- Array("gini", "entropy")
    maxBins <- Array(25, 50, 100, 200, 300)
  } yield {
    val model = DecisionTree.trainClassifier(train, numClasses, categoricalFeatures, impurity, maxDepth, maxBins)
    (model, (impurity, maxBins), evaluateModel(getMetrics(model, validation)).map(_.sum).sum)
  }

  val (model, (impurity, maxBins), _) = evaluations.maxBy(_._3)
  println(s"Selected impurity: $impurity, selected maxBins: $maxBins")
  val metrics = getMetrics(model, test)
  println(metrics.confusionMatrix)
  evaluateModel(metrics).foreach(println)

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example => (model.predict(example.features), example.label))
    new MulticlassMetrics(predictionsAndLabels)
  }
}
