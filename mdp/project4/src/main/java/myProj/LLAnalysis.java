package myProj;

import burlap.behavior.functionapproximation.DifferentiableStateActionValue;
import burlap.behavior.functionapproximation.dense.ConcatenatedObjectFeatures;
import burlap.behavior.functionapproximation.dense.DenseCrossProductFeatures;
import burlap.behavior.functionapproximation.dense.NormalizedVariableFeatures;
import burlap.behavior.functionapproximation.dense.NumericVariableFeatures;
import burlap.behavior.functionapproximation.dense.fourier.FourierBasis;
import burlap.behavior.functionapproximation.dense.rbf.DistanceMetric;
import burlap.behavior.functionapproximation.dense.rbf.RBFFeatures;
import burlap.behavior.functionapproximation.dense.rbf.functions.GaussianRBF;
import burlap.behavior.functionapproximation.dense.rbf.metrics.EuclideanDistance;
import burlap.behavior.functionapproximation.sparse.tilecoding.TileCodingFeatures;
import burlap.behavior.functionapproximation.sparse.tilecoding.TilingArrangement;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.policy.PolicyUtils;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.ArrowActionGlyph;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.LandmarkColorBlendInterpolation;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.PolicyGlyphPainter2D;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.common.StateValuePainter2D;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.valuefunction.QFunction;
import burlap.behavior.valuefunction.ValueFunction;
import burlap.mdp.auxiliary.stateconditiontest.StateConditionTest;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.state.State;
import burlap.mdp.core.state.vardomain.VariableDomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.common.VisualActionObserver;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.FactoredModel;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.mdp.core.oo.OODomain;
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.behavior.functionapproximation.sparse.tilecoding.TileCodingFeatures;
import burlap.behavior.functionapproximation.sparse.tilecoding.TilingArrangement;
import burlap.domain.singleagent.lunarlander.LLVisualizer;
import burlap.domain.singleagent.lunarlander.LunarLanderDomain;
import burlap.domain.singleagent.lunarlander.LunarLanderTF;
import burlap.domain.singleagent.lunarlander.state.LLAgent;
import burlap.domain.singleagent.lunarlander.state.LLBlock;
import burlap.domain.singleagent.lunarlander.state.LLState;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
//import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentQLearning;

import java.awt.*;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;

public class LLAnalysis
{
    LunarLanderDomain ll;
    OOSADomain domain;
    TerminalFunction tf;
    RewardFunction rf;
    StateConditionTest goalCondition;
    State initialState;
    HashableStateFactory hashingFactory;
    SimulatedEnvironment env;
    TileCodingFeatures tilecoding;
    int nTilings;

    public LLAnalysis()
    {
        LunarLanderDomain ll = new LunarLanderDomain();
        ll.addStandardThrustActions();
        //ll.setXmin(0.0);
        //ll.setYmin(0.0);
        //ll.setXmax(10.0);
        //ll.setYmax(10.0);

        domain = ll.generateDomain();
        tf = new LunarLanderTF(domain);
        ll.setTf(tf);
        goalCondition = new TFGoalCondition(tf);

        //rf = new GoalBasedRF(this.goalCondition, 5.0, -0.1);

        LLState initialState = new LLState(new LLAgent(5, 0, 0), new LLBlock.LLPad(75, 95, 0, 10, "pad"));
        hashingFactory = new SimpleHashableStateFactory();

        env = new SimulatedEnvironment(domain, initialState);

        /*
        ConcatenatedObjectFeatures inputFeatures = new ConcatenatedObjectFeatures()
                .addObjectVectorizion(LunarLanderDomain.CLASS_AGENT, new NumericVariableFeatures());

        nTilings = 5;
        double resolution = 10.;

        double xWidth = (ll.getXmax() - ll.getXmin()) / resolution;
        double yWidth = (ll.getYmax() - ll.getYmin()) / resolution;
        double velocityWidth = 2 * ll.getVmax() / resolution;
        double angleWidth = 2 * ll.getAngmax() / resolution;

        TileCodingFeatures tilecoding = new TileCodingFeatures(inputFeatures);
        tilecoding.addTilingsForAllDimensionsWithWidths(
                new double []{xWidth, yWidth, velocityWidth, velocityWidth, angleWidth},
                nTilings,
                TilingArrangement.RANDOM_JITTER);
        */

    }

    public void visualize(String outputpath)
    {
        Visualizer v = LLVisualizer.getVisualizer(ll.getPhysParams());
        new EpisodeSequenceVisualizer(v, domain, outputpath);
    }

    public void valueIteration(String outputPath)
    {
        Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.01, 100);

        final long startTime = System.currentTimeMillis();
        Policy p = planner.planFromState(initialState);
        final long endTime = System.currentTimeMillis();

        Episode ea = PolicyUtils.rollout(p, initialState, domain.getModel());
        ea.write(outputPath + "vi");

        double sum = 0;
        for(Double r : ea.rewardSequence) {
            sum += r;
        }

        System.out.println("Steps to exit: " + ea.actionSequence.size());
        System.out.println("Run time: " + (endTime - startTime) + " milliseconds");
        System.out.println("Reward Sum: " + sum);
    }

    public void policyIteration(String outputPath)
    {
        PolicyIteration pi = new PolicyIteration(domain, 0.99, hashingFactory, 0.001,
                1000, 100);

        final long startTime = System.currentTimeMillis();
        Policy p = pi.planFromState(initialState);
        final long endTime = System.currentTimeMillis();

        Episode ea = PolicyUtils.rollout(p, initialState, domain.getModel());
        ea.write(outputPath + "pi");

        double sum = 0;
        for(Double r : ea.rewardSequence) {
            sum += r;
        }

        System.out.println("Steps to exit: " + ea.actionSequence.size());
        System.out.println("Run time: " + (endTime - startTime) + " milliseconds");
        System.out.println("Reward Sum: " + sum);
    }

    public void qLearning(String outputPath)
    {
        /*
        double defaultQ = 0.5;
        DifferentiableStateActionValue vfa = tilecoding.generateVFA(defaultQ/nTilings);
        GradientDescentQLearning agent = new GradientDescentQLearning(domain, 0.99, vfa, 1.);

        List<Episode> episodes = new ArrayList<Episode>();
        for(int i = 0; i < 5000; i++){
            Episode ea = agent.runLearningEpisode(env);
            episodes.add(ea);
            System.out.println(i + ": " + ea.maxTimeStep());
            env.resetEnvironment();
        }*/

        LearningAgent agent = new QLearning(domain, 0.99, hashingFactory, 0., 1.);

        // run for n episodes
        for (int i = 0; i < 1000; i++)
        {
            Episode e = agent.runLearningEpisode(env);
            e.write(outputPath + "ql_" + i);
            System.out.println(i + ": " + e.maxTimeStep());

            //reset environment for next learning episode
            env.resetEnvironment();
        }
    }

    public void experimentAndPlotter()
    {
        //different reward function for more structured performance plots
        ((FactoredModel) domain.getModel()).setRf(new GoalBasedRF(this.goalCondition, 5.0, -0.1));

        LearningAgentFactory qLearningFactory = new LearningAgentFactory()
        {
            public String getAgentName()
            {
                return "Q-Learning";
            }


            public LearningAgent generateAgent()
            {
                return new QLearning(domain, 0.99, hashingFactory, 0.3, 0.1);
            }
        };

        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(
                env, 10, 1000, qLearningFactory);
        exp.setUpPlottingConfiguration(500, 250, 2, 1000,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                PerformanceMetric.AVERAGE_EPISODE_REWARD);

        exp.startExperiment();
        exp.writeStepAndEpisodeDataToCSV("expDataLL");
    }

    public static void main(String[] args)
    {
        LLAnalysis analysis = new LLAnalysis();
        String outputPath = "output_ll/";

        //analysis.valueIteration(outputPath);
        //analysis.policyIteration(outputPath);
        analysis.qLearning(outputPath);

        //analysis.experimentAndPlotter();
        analysis.visualize(outputPath);
    }
}
