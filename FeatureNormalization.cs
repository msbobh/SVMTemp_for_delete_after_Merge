using Accord.Statistics;


namespace SVMPrototype_accord
{
    internal class FeatureNormalization
    {
        /*
         * Feature scales and and mean normalizes a feature matrix.
         * The values are normalized by subtracting the mean value of each feature from each value
         * in the input matrix.  
         * After subtracting the mean each feature is scaled (divided) by their respective
         * standard deviations. Using STD is an alternative to taking the 
         * range of values using max/min.  
         *  
         */
        public double[][] result;
        public FeatureNormalization(in double[][] inPutMatrix)
        {
            double [] mean = Measures.Mean(inPutMatrix,0);          // Get by Column means
            double [] Sigma = inPutMatrix.StandardDeviation();      // Calculate STD deviations by column   
            double[][] tempMatrix = inPutMatrix;
            for (int row = 0; row < inPutMatrix.Length; row++)
            {
                for (int col = 0; col < inPutMatrix[0].Length; col++)
                {
                    tempMatrix[row][col] = (inPutMatrix [row][col] - mean[col]) / Sigma[col];
                }
                
            }
            result = tempMatrix;
        }

    }
    
}
