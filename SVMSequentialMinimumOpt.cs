using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using SVM_Using_SequentialMinimumOptimization;

namespace SVM_Using_SequentialMinimumOptimization
{
    /*
     * Complexity (cost) parameter C. Increasing the value of C forces the creation of a more accurate model
     * that may not generalize well. If this value is not set and UseComplexityHeuristic is set to true, 
     * the framework will automatically guess a value for C. If this value is manually set to something else,
     * then UseComplexityHeuristic will be automatically disabled and the given value will be used instead. 
     */

    static public class ConvertboolToint
    {
        public static int[] _Convert (in bool[] _input)
        {
             int[] ConvertedResults = new int[_input.Length];
            for (int i = 0; i < _input.Length; i++)
            {
                ConvertedResults[i] = Convert.ToInt32(_input[i]);
            }
            return ConvertedResults;
        }
           
        }

       

    }
    public class SVM_Gaussian
    {
        // Instance variables
        private SupportVectorMachine<Gaussian> _gaussianLearnObject; // Model object for the class

        // constructor
        public SVM_Gaussian(in double[][] inputs, in int[] _xor)
        {
            // Now, we can create the sequential minimal optimization teacher
            SequentialMinimalOptimization<Gaussian> learn = new SequentialMinimalOptimization<Gaussian>()
            {
                UseComplexityHeuristic = true,
                UseKernelEstimation = false
            };

            // And then we can obtain a trained SVM by calling its Learn method
            _gaussianLearnObject = learn.Learn(inputs, _xor);
        }
        // Properties
        public bool[] Prediction(double[][] inputs)
        {
            // Finally, we can obtain the decisions predicted by the machine:
            return _gaussianLearnObject.Decide(inputs);
        }
        public SupportVectorMachine<Gaussian> Model
        {
            get => _gaussianLearnObject;
        }
        // Methods
        public int[] Predict_aka_Decide (in double[][] _inputs)
        {
            bool [] returnbools = _gaussianLearnObject.Decide (_inputs);
            return ConvertboolToint._Convert(returnbools);
        }
        
    }
public class SVM_Sigmoid
{
    // Instance variables
    private SupportVectorMachine<Sigmoid> _SigLearnObject;

    // Constructor
    public SVM_Sigmoid(in double[][] _inputs, in int[] _labels)
    {
        SequentialMinimalOptimization<Sigmoid> learn = new SequentialMinimalOptimization<Sigmoid>()
        {
            UseComplexityHeuristic = true,
            UseKernelEstimation = true
        };
        _SigLearnObject = learn.Learn(_inputs, _labels);
    }
    //**************************************************
    // Properties
    //**************************************************
    public bool[] Predict(in double[][] _inputs)
    {
        return _SigLearnObject.Decide(_inputs);
    }
    public SupportVectorMachine<Sigmoid> Model
    {
        get => _SigLearnObject;
    }
    //**************************************************
    // Methods
    //**************************************************
    public int[] Decide(in double[][] _inputs)
    {
        //return ConvertboolToint._Convert(_SigLearnObject.Decide(_inputs));
        Func<bool[], int[]> conVurt = (bool[] bul) => ConvertboolToint._Convert(bul);
        return conVurt(_SigLearnObject.Decide(_inputs));
    }
}
        

    public class SVM_Poly
    {
        // instance variables
        private double _tolerance = 1.0e-2;
        private double _epsilon = 1.0e-3;
        private SupportVectorMachine<Polynomial> LearnObject;
        private SequentialMinimalOptimization<Polynomial> _kvsm;

        // Constructor for an SVM with Polynomial Kernel
        public SVM_Poly(in double[][] _input, in int[] _labels)
        {
            /*
             * The Complexity property controls how complicated the decision boundary line
             * is allowed to become. Higher values can give better accuracy at the expense of 
             * increased likelihood of model overfitting. A good value for Complexity must be 
             * determined by trial and error. The Epsilon and Tolerance properties also control 
             * the SMO algorithm. The values used, 1.0e-3 and 1.0e-2, are the default values and 
             * so those two statements to set those properties could have been left out.
             * 
             * The Polynomial object constructor accepts a degree argument and an r-constant argument,
             * but does not accept a gamma argument. This effectively sets gamma to a constant value of 
             * 1.0 and is a quirk of the Accord.NET library
             */

            Console.WriteLine("Creating and training Poly kernel SVM");
            _kvsm = new SequentialMinimalOptimization<Polynomial>();
            {
                _kvsm.Complexity = 1.0;
                _kvsm.Kernel = new Polynomial(2, 0.0);
                _kvsm.Epsilon = _epsilon;        // Default value =  1.0e-3;
                _kvsm.Tolerance = _tolerance;    // 1.0e-2;
            }

            // debugging statments to be removed
            Console.WriteLine("Starting training SMOPoly");
            LearnObject = _kvsm.Learn(_input, _labels);
            Console.WriteLine("Training complete. SMOPoly");

        }
        // Properties
        public SupportVectorMachine<Polynomial> Model
        {
            get => LearnObject;
        }
        public double Tolerance
        {
            get { return _tolerance; } 
            set { _tolerance = Tolerance; }
        }
        public double Epsilon
        {
            get { return _epsilon; }
            set { _epsilon = value; }
        }
               
    }


