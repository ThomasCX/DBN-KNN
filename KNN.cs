using System;
namespace KNN
{
  class KNNProgram
  {
    static void Main(string[] args)
    {

      double[][] trainData = LoadData();
     
      int numFeatures = 2;  // predictor variables
      int numClasses = 3;   // 0, 1, 2
      
      double[] unknown = new double[] { 5.25, 1.75 };
      Console.WriteLine("Classifying item with predictor values: 5.25 1.75 \n");

      int k = 1;
      Console.WriteLine("With k = 1");
      int predicted = Classify(unknown, trainData, numClasses, k);
      Console.WriteLine("\nPredicted class = " + predicted);
      Console.WriteLine("");

      k = 4;
      Console.WriteLine("With k = 4");
      predicted = Classify(unknown, trainData, numClasses, k);
      Console.WriteLine("\nPredicted class = " + predicted);
      Console.WriteLine("");

      Console.WriteLine("End k-NN demo \n");
      Console.ReadLine();
    } // Main

    static int Classify(double[] unknown, double[][] trainData, int numClasses, int k)
    {
      int n = trainData.Length;  // number data items
      IndexAndDistance[] info = new IndexAndDistance[n];
      for (int i = 0; i < n; ++i)
      {
        IndexAndDistance curr = new IndexAndDistance();
        double dist = Distance(unknown, trainData[i]);
        curr.idx = i;
        curr.dist = dist;
        info[i] = curr;
      }

      Array.Sort(info);  // sort by distance
      Console.WriteLine("\nNearest  /  Distance  / Class");
      Console.WriteLine("==============================");
      for (int i = 0; i < k; ++i)
      {
        int c = (int)trainData[info[i].idx][2];
        string dist = info[i].dist.ToString("F3");
        Console.WriteLine("( " + trainData[info[i].idx][0] + "," + trainData[info[i].idx][1] + " )  :  " + dist + "        " + c);
      }

      int result = Vote(info, trainData, numClasses, k);  // k nearest classes
      return result;

    } // Classify

    static int Vote(IndexAndDistance[] info, double[][] trainData, int numClasses, int k)
    {
      int[] votes = new int[numClasses];  // one cell per class
      for (int i = 0; i < k; ++i)  // just first k nearest
      {
        int idx = info[i].idx;  // which item
        int c = (int)trainData[idx][2];  // class in last cell
        ++votes[c];
      }

      int mostVotes = 0;
      int classWithMostVotes = 0;
      for (int j = 0; j < numClasses; ++j)
      {
        if (votes[j] > mostVotes)
        {
          mostVotes = votes[j];
          classWithMostVotes = j;
        }
      }

      return classWithMostVotes;
    }

    static double Distance(double[] unknown, double[] data)
    {
      double sum = 0.0;
      for (int i = 0; i < unknown.Length; ++i) 
        sum += (unknown[i] - data[i]) * (unknown[i] - data[i]);
      return Math.Sqrt(sum);
    }

  } // Program

  public class IndexAndDistance : IComparable<IndexAndDistance>
  {
    public int idx;  // index of a training item
    public double dist;  // distance from train item to unknown

    // need to sort these to find k closest
    public int CompareTo(IndexAndDistance other)
    {
      if (this.dist < other.dist) return -1;
      else if (this.dist > other.dist) return +1;
      else return 0;
    }
  }

} // ns
