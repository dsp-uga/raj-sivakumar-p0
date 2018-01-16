import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

val config = new SparkConf().setMaster(" local").setAppName("p0")
val sc = new SparkContext(config)
