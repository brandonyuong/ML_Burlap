package myProj;

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
import burlap.behavior.singleagent.learning.tdmethods.SarsaLam;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.deterministic.DeterministicPlanner;
import burlap.behavior.singleagent.planning.deterministic.informed.Heuristic;
import burlap.behavior.singleagent.planning.deterministic.informed.astar.AStar;
import burlap.behavior.singleagent.planning.deterministic.uninformed.bfs.BFS;
import burlap.behavior.singleagent.planning.deterministic.uninformed.dfs.DFS;
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
import burlap.statehashing.HashableStateFactory;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import burlap.domain.singleagent.frostbite.FrostbiteDomain;
import burlap.domain.singleagent.frostbite.FrostbiteTF;
import burlap.domain.singleagent.frostbite.FrostbiteRF;
import burlap.domain.singleagent.frostbite.FrostbiteVisualizer;
import burlap.domain.singleagent.frostbite.state.FrostbiteState;
import burlap.domain.singleagent.frostbite.state.FrostbitePlatform;
import burlap.domain.singleagent.frostbite.state.FrostbiteIgloo;
import burlap.domain.singleagent.frostbite.state.FrostbiteAgent;

import java.awt.*;
import java.util.List;

public class FbiteAnalysis
{
    FrostbiteDomain fbd;
    OOSADomain domain;
    TerminalFunction tf;
    StateConditionTest goalCondition;
    State initialState;
    HashableStateFactory hashingFactory;
    SimulatedEnvironment env;

    public FbiteAnalysis()
    {
        fbd = new FrostbiteDomain();
        tf = new FrostbiteTF(domain);
        fbd.setTf(tf);
        goalCondition = new TFGoalCondition(tf);
        domain = fbd.generateDomain();

        initialState = new FrostbiteState();
        hashingFactory = new SimpleHashableStateFactory();

        env = new SimulatedEnvironment(domain, initialState);


        //VisualActionObserver observer = new VisualActionObserver(domain,
        //	GridWorldVisualizer.getVisualizer(gwdg.getMap()));
        //observer.initGUI();
        //env.addObservers(observer);
    }

    public void visualize(String outputpath)
    {
        Visualizer v = FrostbiteVisualizer.getVisualizer();
        new EpisodeSequenceVisualizer(v, domain, outputpath);
    }

    public void valueIteration(String outputPath)
    {
        Planner planner = new ValueIteration(domain, 0.99, hashingFactory, 0.001, 100);

        final long startTime = System.currentTimeMillis();
        Policy p = planner.planFromState(initialState);
        final long endTime = System.currentTimeMillis();

        Episode ea = PolicyUtils.rollout(p, initialState, domain.getModel());
        ea.write(outputPath + "vi");

        //simpleValueFunctionVis((ValueFunction) planner, p);
        //manualValueFunctionVis((ValueFunction)planner, p);

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

        //simpleValueFunctionVis((ValueFunction) pi, p);
        //manualValueFunctionVis((ValueFunction)pi, p);

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
        exp.writeStepAndEpisodeDataToCSV("expDataFB");
    }

    public static void main(String[] args)
    {
        FbiteAnalysis analysis = new FbiteAnalysis();
        String outputPath = "outputFB/";

        //analysis.valueIteration(outputPath);
        //analysis.policyIteration(outputPath);
        analysis.qLearning(outputPath);

        //analysis.experimentAndPlotter();

        //analysis.visualize(outputPath);
    }
}
