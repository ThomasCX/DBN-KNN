using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;

namespace DeepLearn
{
    public static class DataManager
    {
        public static double[][] Load(string pathName, out double[][] outputs)
        {
            List<double[]> list = new List<double[]>();
            List<double[]> output = new List<double[]>();
            List<double[]> listTemp = new List<double[]>();

            // Read data file.
            using (FileStream fs = File.Open(pathName, FileMode.Open, FileAccess.Read))
            {
                using (BufferedStream bs = new BufferedStream(fs))
                {
                    using (StreamReader sr = new StreamReader(bs))
                    {
                        List<double> row = new List<double>();

                        bool readOutput = false;

                        string line;
                        while ((line = sr.ReadLine()) != null)
                        {
                            // Collect each 0 and 1 from the data.
                            foreach (char ch in line)
                            {
                                if (!readOutput)
                                {
                                    // Reading input.
                                    if (ch != ' ' && ch != '\n')//如果不等于空格和换行
                                    {
                                        // Add this digit to our input.
                                        row.Add(Double.Parse(ch.ToString()));
                                    }
                                    else if (ch == ' ')//如果等于空格                             
                                    {
                                        // End of input reached. Store the input row.
                                        list.Add(row.ToArray());

                                        // Start a new input row.
                                        row = new List<double>();

                                        // Set flag to read output label.
                                        readOutput = true;
                                    }
                                }
                                else
                                {
                                    // Read output label.
                                    output.Add(FormatOutputVector(Double.Parse(ch.ToString())));

                                    // Set flag to read inputs for next row.
                                    readOutput = false;
                                }
                            }
                        }
                    }
                }
            }

            // Set outputs.
            outputs = output.ToArray();

            //int j = 0;
            //for (int i = 0; i < listTemp.Count; i++)
            //{
            //    var firstArray = listTemp[i].Take(listTemp[i].Length / 2).ToArray();
            //    var secondArray = listTemp[i].Skip(listTemp[i].Length / 2).ToArray();

            //    if (j < 2 * listTemp.Count)
            //    {
            //        list.Add(firstArray);
            //        j++;
            //        list.Add(secondArray);
            //        j++;
            //    }
            //}

            // Return inputs;
            return list.ToArray();
        }

        public static List<List<int>> LoadOri(string pathName)
        {
            int i = 0;
            List<string> list = new List<string>();
            StreamReader myStreamReader = new StreamReader(pathName);
            string line;
            while (i >= 0)
            {
                line = myStreamReader.ReadLine();  //将整个txt切分成多个字符串
                if (line != null)
                {
                    list.Add(line);
                    i++; //i为.txt的行数
                }
                else break;
            }
            char[][] array = new char[i][];
            string[] arrayTemp = list.ToArray();

            for (int x = 0; x < array.Length; x++)
            {
                int count = 0;
                foreach (var charN in arrayTemp[x])
                {
                    count++;  //获取每个字符串中字符的个数
                }
                array[x] = arrayTemp[x].ToCharArray(); //切分每个字符串，将n个字符存入数组中
            }
            //解法一：用多重list方式实现
            List<List<int>> ListOutput = new List<List<int>>();
            for (int x1 = 0; x1 < array.Length; x1++)
            {
                List<int> Ltemp = new List<int>();
                for (int x2 = 0; x2 < array[x1].Length; x2++)
                {
                    string TempString;
                    int TempInt = 0;
                    TempString = Convert.ToString(array[x1][x2]); //将char类型强制转换为string类型
                    TempInt = Convert.ToInt16(TempString);
                    Ltemp.Add(TempInt);
                }
                ListOutput.Add(Ltemp);
            }

            return ListOutput;
        }




        /// <summary>
        /// Converts a numeric output label (0, 1, 2, 3, etc) to its cooresponding array of doubles, where all values are 0 except for the index matching the label (ie., if the label is 2, the output is [0, 0, 1, 0, 0, ...]).
        /// </summary>
        /// <param name="label">double</param>
        /// <returns>double[]</returns>
        public static double[] FormatOutputVector(double label)
        {
            double[] output = new double[10];

            for (int i = 0; i < output.Length; i++)
            {
                if (i == label)
                {
                    output[i] = 1;
                }
                else
                {
                    output[i] = 0;
                }
            }

            return output;
        }

        /// <summary>
        /// Finds the largest output value in an array and returns its index. This allows for sequential classification from the outputs of a neural network (ie., if output at index 2 is the largest, the classification is class "3" (zero-based)).
        /// </summary>
        /// <param name="output">double[]</param>
        /// <returns>double</returns>
        public static double FormatOutputResult(double[] output)
        {
            return output.ToList().IndexOf(output.Max());
        }

    }
}
