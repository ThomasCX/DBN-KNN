using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DeepLearn;

namespace DeepLearn
{
    /// <summary>
    /// Simple Restricted Boltzmann Machine implementation
    /// </summary>
    public class RBM : IRBM
    {
        public event EpochEventHandler EpochEnd;
        public Logger logger;
        public void RaiseEpochEnd(int seq, double err)
        {
            if (EpochEnd != null)
                EpochEnd(this, new EpochEventArgs(seq, err));
        }

        public event EpochEventHandler TrainEnd;

        public void RaiseTrainEnd(int seq, double err)
        {
            if (TrainEnd != null)
                TrainEnd(this, new EpochEventArgs(seq, err));
        } 

        private readonly int m_numHiddenElements;
        private readonly int m_numVisibleElements;
        private readonly double m_learningRate;
        private RealMatrix m_weights; 

        public int NumberOfVisibleElements { get { return m_numVisibleElements; } }
       
        public RBM(int numVisible, int numHidden, double learningRate = 0.1)
        {
            m_numHiddenElements = numHidden;
            m_numVisibleElements = numVisible;
            m_learningRate = learningRate;

            m_weights = 0.1*Distributions.GaussianMatrix(numVisible, numHidden); //随机权，避免局部最优

            // Insert weights for the bias units into the first row and first column.
            m_weights = m_weights.InsertRow(0);
            m_weights = m_weights.InsertCol(0);
        }

        /// <summary>
        /// Get the hidden layer features from a visible layer
        /// ---------
        /// Assuming the RBM has been trained (so that weights for the network have been learned),
        /// run the network on a set of hidden units, to get a sample of the visible units.
        /// 
        /// Parameters
        /// ---------
        /// data: A matrix where each row consists of the states of the hidden units.
        /// 
        /// Returns
        /// ---------
        /// visible_states: A matrix where each row consists of the visible units activated from the hidden
        /// units in the data matrix passed in.
        /// </summary>
        public double[][] GetHiddenLayer(double[][] dataArray)
        {
            var num_examples = dataArray.Length;

            // Create a matrix, where each row is to be the hidden units (plus a bias unit)
            // sampled from a training example.
            var hidden_states = RealMatrix.Ones(num_examples, m_numHiddenElements + 1);

            var data = new RealMatrix(dataArray);
            // Insert bias units of 1 into the first column of data.
            data = data.InsertCol(1); // np.insert(data, 0, 1, axis = 1)

            // Calculate the activations of the hidden units.
            var hiddenActivations = data * m_weights;
            // Calculate the probabilities of turning the hidden units on.
            //激活函数是指的如何把“激活的神经元的特征”通过函数把特征保留并映射出来，这是神经网络能解决非线性问题关键
            //激活函数，就是在神经网络的神经元上运行的函数，负责将神经元的输入映射到输出端。
            var hiddenProbs = ActivationFunctions.Logistic(hiddenActivations);
            // Turn the hidden units on with their specified probabilities.
            //以指定的概率打开隐藏的单元
            hidden_states = hiddenProbs > Distributions.UniformRandromMatrix(num_examples, m_numHiddenElements + 1);

            // Ignore the bias units.
            hidden_states = hidden_states.RemoveFirstCol(); 
            return hidden_states;
        }

        /// <summary>
        /// Get the visible layer from a hidden layer
        /// ---------
        /// Assuming the RBM has been trained (so that weights for the network have been learned),
        /// run the network on a set of visible units, to get a sample of the hidden units.
        /// Parameters
        /// ----------
        /// data: A matrix where each row consists of the states of the visible units.
        /// 
        /// Returns
        /// -------
        /// hidden_states: A matrix where each row consists of the hidden units activated from the visible
        /// units in the data matrix passed in.
        /// </summary>
        public double[][] GetVisibleLayer(double[][] dataArray)
        {
            var numExamples = dataArray.Length;

            // Create a matrix, where each row is to be the visible units (plus a bias unit)
            // sampled from a training example.

            var data = new RealMatrix(dataArray);
            // Insert bias units of 1 into the first column of data.
            data = data.InsertCol(1);

            // Calculate the activations of the visible units.
            var visibleActivations = data * m_weights.Transpose;
            // Calculate the probabilities of turning the visible units on.
            var visibleProbs = ActivationFunctions.Logistic(visibleActivations);
            // Turn the visible units on with their specified probabilities.
            var visibleStates = visibleProbs > Distributions.UniformRandromMatrix(numExamples, m_numVisibleElements + 1);
            // Always fix the bias unit to 1

            // Ignore the bias units.
            visibleStates = visibleStates.RemoveFirstCol(); //visible_states[:,1:]
            return visibleStates;
        }

        /// <summary>
        /// Day dream - Reconstruct a randrom matrix (An interesting way of seeing strong features the machine has learnt).
        /// 
        /// Randomly initialize the visible units once, and start running alternating Gibbs sampling steps
        /// (where each step consists of updating all the hidden units, and then updating all of the visible units),
        /// taking a sample of the visible units at each step.
        /// 
        /// Note that we only initialize the network "once", so these samples are correlated.
        /// ---------
        /// samples: A matrix, where each row is a sample of the visible units produced while the network was daydreaming。
        /// </summary>
        /// <param name="numberOfSamples">How many images/dreams</param>
        /// <returns>Array of Reconstructed dreams</returns>
        public double[][] DayDream(int numberOfSamples)
        {
            //Create a matrix, where each row is to be a sample of of the visible units 
            //(with an extra bias unit), initialized to all ones.
            var data = RealMatrix.Ones(numberOfSamples, m_numVisibleElements + 1);

            //Take the first sample from a uniform distribution.
            data.Update(0, 1, Distributions.UniformRandromMatrixBool(1, m_numVisibleElements), 1);

            //Start the alternating Gibbs sampling.
            //Note that we keep the hidden units binary states, but leave the
            //visible units as real probabilities. 
            //See section 3 of Hinton's "A Practical Guide to Training Restricted Boltzmann Machines" for more on why.
            for (int i = 0; i < numberOfSamples; i++)
            {
                var visible = data.Submatrix(i, 0, 1).ToVector();
                //Calculate the activations of the hidden units.
                var hidden_activations = (visible*m_weights).ToVector();
                //Calculate the probabilities of turning the hidden units on.
                var hidden_probs = ActivationFunctions.Logistic(hidden_activations);
                //Turn the hidden units on with their specified probabilities.
                var hidden_states = hidden_probs > RVector.Random(m_numHiddenElements + 1);
                //Always fix the bias unit to 1.
                hidden_states[0] = 1;

                //Recalculate the probabilities that the visible units are on.
                var visible_activations = (hidden_states*m_weights.Transpose).ToVector();
                var visible_probs = ActivationFunctions.Logistic(visible_activations);
                var visible_states = visible_probs > RVector.Random(m_numVisibleElements + 1);
                data.Update(visible_states, 0, false, i, 0);
            }

            return data.Submatrix(0, 1).ToArray();
        }
    
        //public void AsyncTrain(double[][] data, int maxEpochs)
        //{
        //    double e = 0;
        //    var f = new TaskFactory();
        //    f.StartNew(new Action(() => Train(data, maxEpochs, out e)));
        //}

        //原理参考：http://www.cnblogs.com/pinard/p/6530523.html或者http://deeplearning.net/tutorial/rbm.html
        public void Train(double[][] dataArray, int maxEpochs, out double error)
        {
            error = 0;
            var numExamples = dataArray.Length;
            var data = new RealMatrix(dataArray);

            // Insert bias units of 1 into the first column.
            data = data.InsertCol(1);
            Stopwatch sw = new Stopwatch();//准确地测量运行时间。
            for (int i = 0; i < maxEpochs; i++)
            {
                sw.Start();
                //Clamp to the data and sample from the hidden units. 
                //This is the "positive CD phase", aka the reality phase
                var posHiddenActivations = data * m_weights;
                var posHiddenProbs = ActivationFunctions.Logistic(posHiddenActivations);//隐层神经元被激活的概率
                posHiddenProbs = posHiddenProbs.Update(0, 1, 1); // 相当于P(hj=1|v)，Fix the bias unit

                var posHiddenStates = posHiddenProbs > Distributions.UniformRandromMatrix(numExamples, m_numHiddenElements + 1);
                // Note that we're using the activation "probabilities" of the hidden states, not the hidden states 
                // themselves, when computing associations. We could also use the states; see section 3 of Hinton's 
                // "A Practical Guide to Training Restricted Boltzmann Machines" for more
                var posAssociations = data.Transpose * posHiddenProbs;

                // Reconstruct the visible units and sample again from the hidden units
                // This is the "negative CD phase", aka the daydreaming phase
                var negVisibleActivations = posHiddenStates * m_weights.Transpose;
                var negVisibleProbs = ActivationFunctions.Logistic(negVisibleActivations);//隐藏层到可见层
                negVisibleProbs = negVisibleProbs.Update(0, 1, 1); //相当于P(vj=1|h)

                var negHiddenActivations = negVisibleProbs * m_weights;
                var negHiddenProbs = ActivationFunctions.Logistic(negHiddenActivations);
                // Note, again, that we're using the activation "probabilities" when computing associations, not the states themselves
                var negAssociations = negVisibleProbs.Transpose * negHiddenProbs;
                

                // Update weights
                m_weights = m_weights + (m_learningRate * ((posAssociations - negAssociations) / numExamples));

                sw.Stop(); //计时结束
                error = (((data - negVisibleProbs) ^ 2).Average()) / dataArray.Length; //误差除以size

                RaiseEpochEnd(i, error);//记录每次学习的误差
                Console.WriteLine("Epoch {0}: error is {1}, computation time (ms): {2}", i, error, sw.ElapsedMilliseconds);

                sw.Reset();  //运行时间sw清零
            }
            
            RaiseTrainEnd(maxEpochs, error);
        }
        
        public double[][] Reconstruct(double[][] data)
        {
            var hl = GetHiddenLayer(data);
            return GetVisibleLayer(hl);
        }

        public RealMatrix GetWeights()
        {
            return new RealMatrix(m_weights); //Make a Copy
        }
    }

    public delegate void EpochEventHandler(object sender, EpochEventArgs e);
    public class EpochEventArgs : EventArgs
    {
        public int SequenceNumber;
        public double Error;

        public EpochEventArgs(int sequenceNum, double error)
        {
            SequenceNumber = sequenceNum;
            Error = error;
        }
    }
}
