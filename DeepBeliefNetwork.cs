using System;
using System.Threading.Tasks;
using System.Diagnostics;

namespace DeepLearn
{
    public class DeepBeliefNetwork : IDBN
    {
        private readonly RBM[] m_rbms; 

        public event EpochEventHandler EpochEnd;
        public void RaiseEpochEnd(int seq, double err)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs(seq, err));
        }

        //public event EpochEventHandler TrainEnd;
        public void RaiseTrainEnd(double error)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs(0, error));
        } 

        public DeepBeliefNetwork(int[] layerSizes, double learningRate)
        {
            m_rbms = new RBM[layerSizes.Length - 1];

            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                var rbm = new RBM(layerSizes[i], layerSizes[i + 1], learningRate);
                rbm.EpochEnd += OnRbm_EpochEnd;
                m_rbms[i] = rbm;
            }
        } 

        private void OnRbm_EpochEnd(object sender, EpochEventArgs e)
        {
            RaiseEpochEnd(e.SequenceNumber, e.Error);
        } 

        public double[][] Encode(double[][] data)  //编码
        {
            data = m_rbms[0].GetHiddenLayer(data);

            for (int i = 0; i < m_rbms.Length - 1; i++)
            {
                data = m_rbms[i + 1].GetHiddenLayer(data);
            }
            return data;
        }

        public double[][] Decode(double[][] data)  //解码
        {
            data = m_rbms[m_rbms.Length - 1].GetVisibleLayer(data);

            for (int i = m_rbms.Length - 1; i > 0; i--)
            {
                data = m_rbms[i - 1].GetVisibleLayer(data);
            }

            return data;
        }

        public double[][] Reconstruct(double[][] data)
        {
            var hl = Encode(data);
            var h2 = Decode(hl);
            return h2;
        }

        public double[][] DayDream(int numOfDreams)
        {
            var dreamRawData = Distributions.UniformRandromMatrixBool(numOfDreams, m_rbms[0].NumberOfVisibleElements);
            var ret = Reconstruct(dreamRawData);
            return ret;
        }

        //DBN训练模型；并融合KNN
        //pre-traing+fine-tune
        public void TrainAll(double[][] visibleData, int epochs, int epochMultiplier)  
        {
            //RealMatrix weights;
            double error;
            Stopwatch sw = new Stopwatch();
            sw.Start();
            Stopwatch sw1 = new Stopwatch();        
            for (int i = 0; i < m_rbms.Length; i++)
            {
                sw1.Start();
                m_rbms[i].Train(visibleData, epochs, out error);
                //RaiseTrainEnd(error);
                visibleData = m_rbms[i].GetHiddenLayer(visibleData);
                epochs = epochs * epochMultiplier;
                RaiseTrainEnd(error);
                sw1.Stop(); 
                Console.WriteLine("Round {0}: Whole Computation time (ms): {1}", i, sw1.ElapsedMilliseconds);
                sw1.Reset();  
            }
            sw.Stop(); //计时结束     
            Console.WriteLine("Whole Computation time (ms): {0}", sw.ElapsedMilliseconds-1);
            sw.Reset();  //运行时间sw清零
        }

        //训练第二步；该步得到的GetHiddenFeature，作为第一步的输入
        //public double[][] Train(double[][] data, int epochs, int layerNumber, out double error) 
        //{
        //    m_rbms[layerNumber].Train(data, epochs, out error);
        //    RaiseTrainEnd(error);
        //    var GetHiddenFeature = m_rbms[layerNumber].GetHiddenLayer(data);
        //    return GetHiddenFeature;
        //}

        //public void AsyncTrain(double[][] data, int epochs, int layerNumber)
        //{
        //    double error;
        //    var f = new TaskFactory();
        //    f.StartNew(new Action(() => Train(data, epochs, layerNumber, out error)));
        //}

        //public void AsyncTrainAll(double[][] visibleData, int epochs, int epochMultiplier)
        //{
        //    var f = new TaskFactory();
        //    f.StartNew(() => TrainAll(visibleData, epochs, epochMultiplier));
        //} 
    }
}
