using Accord.MachineLearning.DecisionTrees.Learning;
using Accord.Math.Optimization.Losses;
using System;


namespace Consoleapp
{
    class Program
    {
        static void Main(string[] args)
        {


            // in this example, we will learn a decision tree directly from integer
            // matrices that define the inputs and outputs of our learning problem.

            int[][] inputs =
            {
                new int[] { 0, 0 },
                new int[] { 0, 1 },
                new int[] { 1, 0 },
                new int[] { 1, 1 },
            };

            int[] outputs = // xor between inputs[0] and inputs[1]
            {
                0, 1, 1, 0
            };

            // create an id3 learning algorithm
            ID3Learning teacher = new ID3Learning();

            // learn a decision tree for the xor problem
            var tree = teacher.Learn(inputs, outputs);

            // compute the error in the learning
            double error = new ZeroOneLoss(outputs).Loss(tree.Decide(inputs));

            // the tree can now be queried for new examples:
            int[] predicted = tree.Decide(inputs); // should be { 0, 1, 1, 0 }


            /*********************************/

            C45Learning teacher1 = new C45Learning();


        }
    }
}
